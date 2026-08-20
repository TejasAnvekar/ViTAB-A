#!/usr/bin/env python3
"""
VisualCite Attribution Benchmark Runner

Benchmarks Qwen3-VL models on cell attribution task across:
- Multiple model sizes (2B, 4B, 8B, 30B)
- Multiple data representations (JSON, Markdown, Images)
- Multiple prompting strategies (Zero-shot, Few-shot, Chain-of-Thought)
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from config import (
    BenchmarkConfig, 
    DataRepresentation, 
    PromptStrategy,
    ModelSize
)
from data_loader import (
    VisualCiteDataset,
    VisualCiteSample,
    parse_model_output,
    parse_cell_coordinates,
    extract_cells_from_formulas,
    get_json_table_as_readable
)
from model_handler import ModelManager, InferenceResult
from prompt_builder import PromptBuilder
from metrics import (
    evaluate_single_prediction,
    aggregate_metrics,
    CellMetrics,
    AggregatedMetrics,
    format_metrics_table
)
from checkpoint_manager import CheckpointManager, BenchmarkProgress
from result_logger import ResultLogger, create_instance_log, InstanceLog, VCInstanceLog, create_vc_instance_log
from confidence_extractor import extract_cell_confidences, compute_aggregate_confidence, compute_all_cell_probabilities
from verbalized_certainty import VerbalizedCertaintyExtractor
from confidence_types import CellConfidenceResult
from alignment_metrics import compute_cell_alignment_metrics, alignment_metrics_to_dict
from table_utils import get_all_table_cells, get_cell_values

from uncertainty_quantification import run_split_conformal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset: Optional[VisualCiteDataset] = None
        self.model_manager: Optional[ModelManager] = None
        self.prompt_builder = PromptBuilder(jsonl_path=config.jsonl_path)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.result_logger = ResultLogger(config.output_dir)
        self.all_results: List[Dict[str, Any]] = []
        self._cached_certainty_extractor: Optional[VerbalizedCertaintyExtractor] = None
        self._cached_certainty_extractor_model_name: Optional[str] = None
    
    def setup(self) -> None:
        """Initialize dataset and model manager"""
        logger.info("Setting up benchmark...")
        
        # Load dataset
        self.dataset = VisualCiteDataset(
            jsonl_path=self.config.jsonl_path,
            split=self.config.dataset_split,
            max_samples=self.config.max_samples,
            single_cell_only=self.config.single_cell_only
        )
        self.dataset.load()
        
        # Initialize model manager
        self.model_manager = ModelManager(self.config)
        
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        logger.info(f"Models to test: {len(self.config.models)}")
        logger.info(f"Representations: {len(self.config.representations)}")
        logger.info(f"Strategies: {len(self.config.strategies)}")
    
    def run_single_inference(
        self,
        model: Any,
        sample: VisualCiteSample,
        representation: DataRepresentation,
        strategy: PromptStrategy,
        model_name: str,
        extract_confidence: bool = True
    ) -> tuple[InferenceResult, List[str], str, Dict[str, Any]]:
        """
        Run a single inference and return result with parsed predictions.
        
        Args:
            model: The model to use
            sample: The sample to process
            representation: Data representation format
            strategy: Prompting strategy
            model_name: Name of the model being used
            extract_confidence: Whether to extract confidence scores
        
        Returns:
            (InferenceResult, predicted_cells, prompt_used, confidence_data)
            where confidence_data contains internal_confidence and verbalized_certainty
        """
        # Get table representation
        is_image = representation.value.startswith("image_")
        table_content = self.dataset.get_table_representation(sample, representation.value)
        
        # Build prompt (returns tuple with optional example image for few-shot)
        prompt, example_image = self.prompt_builder.build_prompt(
            sample=sample,
            strategy=strategy,
            representation=representation,
            table_content=table_content
        )
        
        # Run inference with logits (always needed for confidence measurement and UQ)
        if is_image:
            result = model.generate_with_image(
                prompt=prompt, 
                image=table_content, 
                example_image=example_image,
                return_logits=True
            )
        else:
            result = model.generate_text_only(prompt=prompt, return_logits=True)
        
        # For CoT, we might need a follow-up extraction
        output_text = result.output_text
        
        if strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            # Check if we need to extract final answer
            if not any(c.isalpha() and c.isupper() for c in output_text[-50:]):
                # No clear cell coordinates at the end, try extraction
                extraction_prompt = self.prompt_builder.build_cot_extraction_prompt(output_text)
                # Note: CoT extraction doesn't use example images
                if is_image:
                    extraction_result = model.generate_with_image(
                        prompt=extraction_prompt, 
                        image=table_content,
                        example_image=None,
                        return_logits=True
                    )
                else:
                    extraction_result = model.generate_text_only(prompt=extraction_prompt, return_logits=True)
                output_text = output_text + "\n\nATTRIBUTED CELLS: " + extraction_result.output_text
                result.inference_time_ms += extraction_result.inference_time_ms
        
        # Parse predicted cells
        predicted_cells = parse_model_output(output_text)
        
        # Extract confidence scores
        confidence_data: Dict[str, Any] = {
            'internal_confidence': None,
            'verbalized_certainty': None,
            'cell_confidences': {},
            'all_cells': [],
            'verbalized_result': None,
            'confidence_query_time_ms': 0.0
        }
        
        if extract_confidence:
            try:
                all_cells = get_all_table_cells(sample.table_json)
                cell_values = get_cell_values(sample.table_json)
                confidence_data['all_cells'] = all_cells
                logger.debug(f"Extracted {len(all_cells)} cells from table for confidence computation")

                # Option-level probabilities over *all* cells (used for conformal UQ)
                if self.config.enable_conformal_uq and result.logits is not None:
                    all_cell_probs = compute_all_cell_probabilities(
                        logits=result.logits,
                        all_cells=all_cells,
                        tokenizer=model.processor.tokenizer,
                        average_across_tokens=True,
                        generated_ids=result.generated_token_ids,
                        predicted_cell=predicted_cells[0] if predicted_cells else None
                    )
                    logger.debug(f"Computed probabilities for {len(all_cell_probs)} cells")
                    # Reduce checkpoint/log size a bit
                    confidence_data['all_cell_probabilities'] = {
                        k: round(float(v), 8) for k, v in all_cell_probs.items()
                    }
                elif self.config.enable_conformal_uq:
                    logger.warning(f"Conformal UQ enabled but logits are None for sample {sample.id}")

                # Internal confidence from first-token logits
                if self.config.extract_internal_confidence and result.logits is not None:
                    if 'all_cell_probabilities' in confidence_data:
                        cell_confidences = {
                            c: confidence_data['all_cell_probabilities'].get(str(c).strip().upper().lstrip("="), 0.0)
                            for c in predicted_cells
                        }
                    else:
                        cell_confidences = extract_cell_confidences(
                            logits=result.logits,
                            all_cells=all_cells,
                            predicted_cells=predicted_cells,
                            tokenizer=model.processor.tokenizer,
                            average_across_tokens=True,
                            generated_ids=result.generated_token_ids
                        )
                    confidence_data['cell_confidences'] = cell_confidences
                    if cell_confidences:
                        confidence_data['internal_confidence'] = compute_aggregate_confidence(cell_confidences)
                        logger.debug(f"Internal confidence: {confidence_data['internal_confidence']:.4f}")
                    else:
                        logger.warning(f"No cell confidences computed for sample {sample.id}")

                # Verbalized certainty (CQP)
                if self.config.extract_verbalized_certainty:
                    import time as _time

                    if self._cached_certainty_extractor is None or self._cached_certainty_extractor_model_name != model.model_name:
                        self._cached_certainty_extractor = VerbalizedCertaintyExtractor(
                            model=model.model,
                            processor=model.processor,
                            max_new_tokens=self.config.cqp_max_tokens,
                            temperature=self.config.cqp_temperature,
                            do_sample=False
                        )
                        self._cached_certainty_extractor_model_name = model.model_name

                    image_for_cqp = table_content if is_image else None

                    # Use the *same* representation content used for the original inference.
                    if not is_image and isinstance(table_content, str):
                        table_for_cqp = table_content
                    else:
                        # For image representations, don't pass table text (image will be used instead)
                        table_for_cqp = None

                    cqp_start = _time.perf_counter()
                    verbalized_result = self._cached_certainty_extractor.query_confidence(
                        question=sample.question,
                        answer=sample.answer,
                        predicted_cells=predicted_cells,
                        table=table_for_cqp,
                        all_cells=all_cells,
                        image=image_for_cqp
                    )
                    query_time_ms = (_time.perf_counter() - cqp_start) * 1000
                    confidence_data['confidence_query_time_ms'] = query_time_ms
                    confidence_data['verbalized_result'] = verbalized_result
                    confidence_data['verbalized_certainty'] = verbalized_result.certainty_score
                    logger.debug(f"Verbalized certainty: {verbalized_result.certainty_score} (level: {verbalized_result.certainty_level.value if verbalized_result.certainty_level else 'None'})")
                    
                    # Log verbalized certainty query separately
                    vc_log = create_vc_instance_log(
                        sample_id=sample.id,
                        model_name=model_name,
                        representation=representation.value,
                        strategy=strategy.value,
                        question=sample.question,
                        answer=sample.answer,
                        predicted_cells=predicted_cells,
                        verbalized_result=verbalized_result,
                        query_time_ms=query_time_ms
                    )
                    self.result_logger.log_vc_instance(vc_log)
                    
            except Exception as e:
                logger.warning(f"Failed to extract internal confidence for sample {sample.id}: {e}")
        
        return result, predicted_cells, prompt, confidence_data
    
    def run_benchmark_configuration(
        self,
        model_name: str,
        representation: DataRepresentation,
        strategy: PromptStrategy
    ) -> AggregatedMetrics:
        """
        Run benchmark for a specific configuration.
        Supports resumption from checkpoints.
        """
        rep_str = representation.value
        strat_str = strategy.value
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {model_name.split('/')[-1]} | {rep_str} | {strat_str}")
        logger.info(f"{'='*60}")
        
        # Check for existing progress
        completed_ids = set()
        progress: Optional[BenchmarkProgress] = None
        existing_metrics: List[CellMetrics] = []
        
        if self.config.resume_from_checkpoint:
            progress = self.checkpoint_manager.load_progress(model_name, rep_str, strat_str)
            if progress is not None:
                completed_ids = set(progress.completed_sample_ids)
                logger.info(f"Resuming from checkpoint: {len(completed_ids)} samples already completed")
                
                # Reconstruct metrics from saved results
                for r in progress.results:
                    existing_metrics.append(CellMetrics(
                        predicted_cells=r['predicted_cells'],
                        ground_truth_cells=r['ground_truth_cells'],
                        cell_precision=r.get('cell_precision', 0),
                        cell_recall=r.get('cell_recall', 0),
                        cell_f1=r.get('cell_f1', 0),
                        row_precision=r.get('row_precision', 0),
                        row_recall=r.get('row_recall', 0),
                        row_f1=r.get('row_f1', 0),
                        col_precision=r.get('col_precision', 0),
                        col_recall=r.get('col_recall', 0),
                        col_f1=r.get('col_f1', 0),
                        exact_match=r.get('exact_match', False),
                        partial_match=r.get('partial_match', False)
                    ))
        
        # Create or reuse progress tracker
        if progress is None:
            progress = self.checkpoint_manager.create_progress(
                model_name=model_name,
                representation=rep_str,
                strategy=strategy.value,
                total_samples=len(self.dataset)
            )
        
        # Load model
        model = self.model_manager.get_model(model_name)
        
        # Prepare samples to process
        samples_to_process = [
            s for s in self.dataset 
            if s.id not in completed_ids
        ]
        
        logger.info(f"Processing {len(samples_to_process)} remaining samples...")
        
        # Run inference
        metrics_list = existing_metrics.copy()
        start_time = time.time()
        checkpoint_counter = 0
        cell_confidence_results: List[CellConfidenceResult] = []

        # If resuming, try to reconstruct confidence results from checkpoint data.
        # This enables confidence-alignment metrics even when there are 0 remaining samples.
        if progress is not None and progress.results:
            try:
                samples_by_id = {s.id: s for s in self.dataset}
                restored = 0
                missing_conf = 0
                for r in progress.results:
                    sid = r.get('sample_id') or r.get('id')
                    if not sid:
                        continue
                    internal = r.get('internal_confidence')
                    verbal = r.get('verbalized_certainty')
                    if internal is None or verbal is None:
                        missing_conf += 1
                        continue
                    sample = samples_by_id.get(sid)
                    if sample is None:
                        continue
                    all_cells = get_all_table_cells(sample.table_json)
                    cell_confidence_results.append(
                        CellConfidenceResult(
                            sample_id=sid,
                            question=sample.question,
                            answer=sample.answer,
                            all_cells=all_cells,
                            predicted_cells=r.get('predicted_cells', []) or [],
                            cell_confidences=r.get('cell_confidences', {}) or {},
                            ground_truth_cells=r.get('ground_truth_cells', []) or [],
                            aggregate_internal_confidence=internal,
                            aggregate_verbalized_certainty=verbal,
                            verbalized_result=None,
                            cell_precision=r.get('cell_precision', 0.0),
                            cell_recall=r.get('cell_recall', 0.0),
                            cell_f1=r.get('cell_f1', 0.0),
                            exact_match=r.get('exact_match', False),
                            inference_time_ms=r.get('inference_time_ms', 0.0),
                            confidence_query_time_ms=0.0,
                            model_name=model_name,
                            representation=rep_str,
                        )
                    )
                    restored += 1

                if restored > 0:
                    logger.info(f"Restored {restored} confidence records from checkpoint for alignment metrics")
                if missing_conf > 0 and (self.config.extract_internal_confidence or self.config.extract_verbalized_certainty):
                    logger.warning(
                        f"{missing_conf} checkpoint results are missing internal/verbalized confidence. "
                        "If you want alignment metrics for these, rerun with --clear-checkpoints (or --no-resume)."
                    )
            except Exception as e:
                logger.warning(f"Failed to restore confidence data from checkpoint: {e}")
        
        for sample in tqdm(samples_to_process, desc=f"{model_name.split('/')[-1][:10]}|{rep_str[:8]}|{strat_str[:8]}"):
            try:
                # Run inference with confidence extraction (optional)
                result, predicted_cells, prompt, confidence_data = self.run_single_inference(
                    model=model,
                    sample=sample,
                    representation=representation,
                    strategy=strategy,
                    model_name=model_name,
                    extract_confidence=(
                        self.config.extract_internal_confidence
                        or self.config.extract_verbalized_certainty
                        or self.config.enable_conformal_uq
                    )
                )
                
                # Get ground truth from answer_formulas
                ground_truth_cells = extract_cells_from_formulas(sample.answer_formulas) if sample.answer_formulas else parse_cell_coordinates(sample.highlighted_cells)
                
                # Evaluate
                cell_metrics = evaluate_single_prediction(predicted_cells, ground_truth_cells)
                metrics_list.append(cell_metrics)
                
                # Store confidence result for alignment analysis
                if self.config.extract_internal_confidence or self.config.extract_verbalized_certainty:
                    cell_confidence_results.append(
                        CellConfidenceResult(
                            sample_id=sample.id,
                            question=sample.question,
                            answer=sample.answer,
                            all_cells=confidence_data.get('all_cells', []),
                            predicted_cells=predicted_cells,
                            cell_confidences=confidence_data.get('cell_confidences', {}) or {},
                            ground_truth_cells=ground_truth_cells,
                            aggregate_internal_confidence=confidence_data.get('internal_confidence'),
                            aggregate_verbalized_certainty=confidence_data.get('verbalized_certainty'),
                            verbalized_result=confidence_data.get('verbalized_result'),
                            cell_precision=cell_metrics.cell_precision,
                            cell_recall=cell_metrics.cell_recall,
                            cell_f1=cell_metrics.cell_f1,
                            exact_match=cell_metrics.exact_match,
                            inference_time_ms=result.inference_time_ms,
                            confidence_query_time_ms=float(confidence_data.get('confidence_query_time_ms') or 0.0),
                            model_name=model_name,
                            representation=rep_str,
                        )
                    )
                
                # Create instance log
                instance_log = create_instance_log(
                    sample_id=sample.id,
                    model_name=model_name,
                    representation=rep_str,
                    strategy=strat_str,
                    question=sample.question,
                    answer=sample.answer,
                    ground_truth_cells=ground_truth_cells,
                    predicted_cells=predicted_cells,
                    raw_output=result.output_text,
                    input_prompt=prompt[:2000],  # Truncate for logging
                    inference_time_ms=result.inference_time_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cell_metrics={
                        'precision': cell_metrics.cell_precision,
                        'recall': cell_metrics.cell_recall,
                        'f1': cell_metrics.cell_f1,
                        'exact_match': cell_metrics.exact_match,
                        'partial_match': cell_metrics.partial_match
                    },
                    confidence_data=confidence_data
                )
                
                # Log instance
                self.result_logger.log_instance(instance_log)
                
                # Update checkpoint
                result_data = {
                    'sample_id': sample.id,
                    'predicted_cells': predicted_cells,
                    'ground_truth_cells': ground_truth_cells,
                    'cell_precision': cell_metrics.cell_precision,
                    'cell_recall': cell_metrics.cell_recall,
                    'cell_f1': cell_metrics.cell_f1,
                    'row_precision': cell_metrics.row_precision,
                    'row_recall': cell_metrics.row_recall,
                    'row_f1': cell_metrics.row_f1,
                    'col_precision': cell_metrics.col_precision,
                    'col_recall': cell_metrics.col_recall,
                    'col_f1': cell_metrics.col_f1,
                    'exact_match': cell_metrics.exact_match,
                    'partial_match': cell_metrics.partial_match,
                    'inference_time_ms': result.inference_time_ms,
                    'internal_confidence': confidence_data.get('internal_confidence'),
                    'verbalized_certainty': confidence_data.get('verbalized_certainty'),
                    'cell_confidences': confidence_data.get('cell_confidences', {}),
                    'all_cell_probabilities': confidence_data.get('all_cell_probabilities')
                }
                
                checkpoint_counter += 1
                if checkpoint_counter >= self.config.checkpoint_every_n_samples:
                    self.checkpoint_manager.add_result(progress, sample.id, result_data)
                    checkpoint_counter = 0
                else:
                    progress.completed_sample_ids.append(sample.id)
                    progress.results.append(result_data)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                continue
        
        # Final checkpoint save
        self.checkpoint_manager.save_progress(progress)
        
        # Aggregate metrics
        total_time = time.time() - start_time
        agg_metrics = aggregate_metrics(metrics_list)
        
        # Log results
        logger.info(format_metrics_table(agg_metrics))
        
        # Confidence alignment summary (Confidence Under the Hood)
        # Always include this in the saved summary when verbalized certainty is enabled.
        alignment_summary: Optional[Dict[str, Any]] = None
        if self.config.extract_verbalized_certainty:
            if cell_confidence_results:
                try:
                    logger.info(f"Computing alignment metrics for {len(cell_confidence_results)} samples...")
                    alignment_metrics = compute_cell_alignment_metrics(cell_confidence_results)
                    alignment_summary = alignment_metrics_to_dict(alignment_metrics)
                    logger.info(
                        f"Alignment: Spearman ρ={alignment_metrics.spearman_rho:.4f} (p={alignment_metrics.spearman_p_value:.4f}), "
                        f"n_valid={alignment_metrics.n_valid_samples}/{alignment_metrics.n_samples}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute alignment metrics: {e}")
                    alignment_summary = {
                        "status": "error",
                        "error": str(e),
                        "n_samples": len(cell_confidence_results),
                    }
            else:
                alignment_summary = {
                    "status": "not_available",
                    "reason": "No confidence records collected for this configuration. This can happen if resuming from a checkpoint that does not store confidence fields, or if extraction was disabled.",
                    "n_samples": 0,
                    "n_valid_samples": 0,
                }
        
        # Save summary
        uq_summary = None
        if self.config.enable_conformal_uq:
            try:
                # Build instances from progress results (includes resumed samples)
                uq_instances = []
                for r in progress.results:
                    if r.get('all_cell_probabilities') and r.get('ground_truth_cells'):
                        uq_instances.append({
                            'sample_id': r.get('sample_id'),
                            'ground_truth_cells': r.get('ground_truth_cells'),
                            'all_cell_probabilities': r.get('all_cell_probabilities'),
                        })

                if len(uq_instances) >= 2:
                    uq_res = run_split_conformal(
                        instances=uq_instances,
                        calibration_ratio=self.config.conformal_calibration_ratio,
                        alpha=self.config.conformal_alpha,
                        seed=self.config.conformal_seed,
                    )
                    uq_summary = {
                        'LAC': uq_res['LAC'].__dict__,
                        'APS': uq_res['APS'].__dict__,
                    }
                    self.result_logger.save_uncertainty_results(
                        model_name=model_name,
                        representation=rep_str,
                        strategy=strat_str,
                        uq_summary=uq_summary,
                    )
                    logger.info(
                        f"Conformal UQ: LAC coverage={uq_res['LAC'].coverage:.3f}, avg_set={uq_res['LAC'].avg_set_size:.2f}; "
                        f"APS coverage={uq_res['APS'].coverage:.3f}, avg_set={uq_res['APS'].avg_set_size:.2f}"
                    )
                else:
                    logger.warning(
                        "Conformal UQ enabled but not enough usable instances. "
                        "If resuming from an old checkpoint, clear checkpoints and rerun to populate all-cell probabilities."
                    )
            except Exception as e:
                logger.warning(f"Failed conformal UQ computation: {e}")

        extra_payload: Optional[Dict[str, Any]] = None
        if uq_summary is not None or alignment_summary is not None:
            extra_payload = {}
            if uq_summary is not None:
                extra_payload["conformal_uq"] = uq_summary
            if alignment_summary is not None:
                extra_payload["confidence_alignment"] = alignment_summary

        # Average main-inference latency across all samples (includes resumed results).
        avg_inference_time_ms = 0.0
        if progress is not None and getattr(progress, "results", None):
            times = []
            for r in progress.results:
                inference_time = r.get("inference_time_ms")
                if inference_time is not None:
                    try:
                        times.append(float(inference_time))
                    except (ValueError, TypeError):
                        continue
            if times:
                avg_inference_time_ms = sum(times) / len(times)
        
        # Fallback: use cell_confidence_results if available
        if avg_inference_time_ms == 0.0 and cell_confidence_results:
            times = [ccr.inference_time_ms for ccr in cell_confidence_results if ccr.inference_time_ms and ccr.inference_time_ms > 0]
            if times:
                avg_inference_time_ms = sum(times) / len(times)

        summary_path = self.result_logger.save_aggregated_results(
            model_name=model_name,
            representation=rep_str,
            strategy=strat_str,
            metrics=agg_metrics,
            total_time_seconds=total_time,
            avg_inference_time_ms=avg_inference_time_ms,
            extra=extra_payload
        )
        
        # Store for final report
        self.all_results.append({
            'run_id': self.result_logger.run_id,
            'model_name': model_name,
            'representation': rep_str,
            'strategy': strat_str,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_time, 2),
            'metrics': {
                'total_samples': agg_metrics.total_samples,
                'cell_level': {
                    'precision': round(agg_metrics.mean_cell_precision, 4),
                    'recall': round(agg_metrics.mean_cell_recall, 4),
                    'f1': round(agg_metrics.mean_cell_f1, 4)
                },
                'row_level': {
                    'precision': round(agg_metrics.mean_row_precision, 4),
                    'recall': round(agg_metrics.mean_row_recall, 4),
                    'f1': round(agg_metrics.mean_row_f1, 4)
                },
                'column_level': {
                    'precision': round(agg_metrics.mean_col_precision, 4),
                    'recall': round(agg_metrics.mean_col_recall, 4),
                    'f1': round(agg_metrics.mean_col_f1, 4)
                },
                'exact_match_rate': round(agg_metrics.exact_match_rate, 4),
                'partial_match_rate': round(agg_metrics.partial_match_rate, 4)
            }
        })
        
        return agg_metrics
    
    def run_full_benchmark(self) -> None:
        """Run the complete benchmark across all configurations"""
        logger.info("Starting full benchmark run...")
        
        total_configs = (
            len(self.config.models) * 
            len(self.config.representations) * 
            len(self.config.strategies)
        )
        logger.info(f"Total configurations to test: {total_configs}")
        
        config_num = 0
        for model_name in self.config.models:
            for representation in self.config.representations:
                for strategy in self.config.strategies:
                    config_num += 1
                    logger.info(f"\n[{config_num}/{total_configs}] Starting configuration...")
                    
                    try:
                        self.run_benchmark_configuration(
                            model_name=model_name,
                            representation=representation,
                            strategy=strategy
                        )
                    except Exception as e:
                        logger.error(f"Error in configuration: {e}")
                        continue
        
        # Generate final outputs
        self.finalize()
    
    def finalize(self) -> None:
        """Generate final reports and exports"""
        logger.info("\nGenerating final outputs...")
        
        # Export results to CSV
        csv_path = self.result_logger.export_all_results_csv(self.all_results)
        logger.info(f"Results CSV: {csv_path}")
        
        # Export detailed instances
        detailed_csv = self.result_logger.export_detailed_instances_csv()
        logger.info(f"Detailed instances CSV: {detailed_csv}")
        
        # Export verbalized certainty logs
        vc_csv = self.result_logger.export_vc_instances_csv()
        if vc_csv:
            logger.info(f"Verbalized certainty CSV: {vc_csv}")
        
        # Generate markdown report
        report_path = self.result_logger.generate_report(self.all_results)
        logger.info(f"Report: {report_path}")
        
        # Cleanup model
        if self.model_manager:
            self.model_manager.cleanup()
        
        logger.info("\nBenchmark complete!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="VisualCite Attribution Benchmark for Qwen3-VL Models"
    )
    
    parser.add_argument(
        '--models', 
        nargs='+',
        default=None,
        help='Model names to benchmark (default: all Qwen3-VL variants)'
    )
    
    parser.add_argument(
        '--representations',
        nargs='+',
        choices=['json', 'markdown', 'image_arial', 'image_times_new_roman', 
                 'image_red', 'image_blue', 'image_green'],
        default=None,
        help='Data representations to test (default: all)'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['zero_shot', 'few_shot', 'chain_of_thought'],
        default=None,
        help='Prompting strategies to test (default: all)'
    )
    
    parser.add_argument(
        '--jsonl-path',
        type=str,
        default='../visualcite.jsonl',
        help='Path to the VisualCite JSONL dataset file (default: ../visualcite.jsonl)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=500,
        help='Maximum samples to process (default: 500)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['train', 'validation', 'dev', 'test'],
        help='Dataset split to use (default: dev)'
    )

    parser.add_argument(
        '--single-cell-only',
        action='store_true',
        help='Only process samples with exactly one ground truth cell'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for checkpoints'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoints'
    )
    
    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear all checkpoints before running'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Torch dtype (default: bfloat16)'
    )
    
    parser.add_argument(
        '--no-flash-attention',
        action='store_true',
        help='Disable flash attention'
    )

    # Confidence extraction
    # NOTE: defaults are defined in BenchmarkConfig; CLI should only override when explicitly provided.
    parser.add_argument(
        '--extract-internal-confidence',
        dest='extract_internal_confidence',
        action='store_const',
        const=True,
        default=None,
        help='Enable internal confidence extraction from first-token logits'
    )

    parser.add_argument(
        '--extract-verbalized-certainty',
        dest='extract_verbalized_certainty',
        action='store_const',
        const=True,
        default=None,
        help='Enable follow-up CQP to get verbalized certainty'
    )

    parser.add_argument(
        '--cqp-template',
        type=str,
        default=None,
        choices=['cell_attribution', 'standard', 'simple', 'cell_multiple_choice'],
        help='CQP template to use (overrides config)'
    )

    parser.add_argument(
        '--cqp-max-tokens',
        type=int,
        default=None,
        help='Max new tokens for the CQP response (overrides config)'
    )

    parser.add_argument(
        '--cqp-temperature',
        type=float,
        default=None,
        help='Temperature for the CQP response (overrides config)'
    )

    # Conformal uncertainty quantification
    parser.add_argument(
        '--enable-conformal-uq',
        action='store_true',
        help='Enable split conformal prediction UQ using first-token logits'
    )

    parser.add_argument(
        '--conformal-calibration-ratio',
        type=float,
        default=None,
        help='Fraction of samples used for calibration (default: 0.5)'
    )

    parser.add_argument(
        '--conformal-alpha',
        type=float,
        default=None,
        help='Miscoverage level alpha (default: 0.1)'
    )

    parser.add_argument(
        '--conformal-seed',
        type=int,
        default=None,
        help='Random seed for calibration/test split (default: 42)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Update logging level if specified
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.setLevel(getattr(logging, args.log_level))
    
    # Build config
    config = BenchmarkConfig(
        jsonl_path=args.jsonl_path,
        dataset_split=args.split,
        max_samples=args.max_samples,
        single_cell_only=args.single_cell_only,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=not args.no_resume,
        device=args.device,
        torch_dtype=args.dtype,
        use_flash_attention=not args.no_flash_attention
    )

    # Confidence settings (only override when explicitly provided)
    if args.extract_internal_confidence is not None:
        config.extract_internal_confidence = args.extract_internal_confidence
    if args.extract_verbalized_certainty is not None:
        config.extract_verbalized_certainty = args.extract_verbalized_certainty
    if args.cqp_template is not None:
        config.cqp_template = args.cqp_template
    if args.cqp_max_tokens is not None:
        config.cqp_max_tokens = args.cqp_max_tokens
    if args.cqp_temperature is not None:
        config.cqp_temperature = args.cqp_temperature

    # Conformal UQ settings
    if args.enable_conformal_uq:
        config.enable_conformal_uq = True
    if args.conformal_calibration_ratio is not None:
        config.conformal_calibration_ratio = args.conformal_calibration_ratio
    if args.conformal_alpha is not None:
        config.conformal_alpha = args.conformal_alpha
    if args.conformal_seed is not None:
        config.conformal_seed = args.conformal_seed
    
    # Override models if specified
    if args.models:
        config.models = args.models
    
    # Override representations if specified
    if args.representations:
        config.representations = [
            DataRepresentation(r) for r in args.representations
        ]
    
    # Override strategies if specified
    if args.strategies:
        config.strategies = [
            PromptStrategy(s) for s in args.strategies
        ]
    
    # Initialize runner
    runner = BenchmarkRunner(config)
    
    # Clear checkpoints if requested
    if args.clear_checkpoints:
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        checkpoint_manager.clear_all_checkpoints()
    
    # Setup and run
    runner.setup()
    runner.run_full_benchmark()


if __name__ == "__main__":
    main()