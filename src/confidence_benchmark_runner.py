"""
Confidence Benchmark Runner for VisualCite Cell Attribution Task.

This module orchestrates the complete Confidence-Probability Alignment
evaluation pipeline, implementing the methodology from the paper
"Confidence Under the Hood" (ACL 2024).

The pipeline:
1. For each sample, present cell attribution as multiple-choice question
2. Extract internal confidence from token probabilities during answer generation
3. Query verbalized certainty using Confidence Querying Prompt (CQP)
4. Compute alignment metrics (Spearman's ρ) between internal and verbalized confidence
5. Analyze relationship between confidence and correctness
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
from PIL import Image

from .config import BenchmarkConfig, DataRepresentation, PromptStrategy
from .data_loader import (
    VisualCiteDataset,
    VisualCiteSample,
    parse_model_output,
    get_json_table_as_readable,
)
from .model_handler import ModelManager, InferenceResult
from .metrics import evaluate_single_prediction, CellMetrics
from .confidence_types import (
    ConfidenceResult,
    CellConfidenceResult,
    AlignmentMetrics,
    InternalConfidenceResult,
    VerbalizedCertaintyResult,
    AlignmentType,
)
from .confidence_extractor import (
    InternalConfidenceExtractor,
    extract_cell_confidences,
    compute_aggregate_confidence,
)
from .verbalized_certainty import (
    VerbalizedCertaintyExtractor,
)
from .alignment_metrics import (
    compute_alignment_metrics,
    compute_cell_alignment_metrics,
    format_alignment_report,
    alignment_metrics_to_dict,
)
from .table_utils import get_all_table_cells, get_cell_values

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBenchmarkConfig(BenchmarkConfig):
    """Extended configuration for confidence benchmark."""
    # Confidence extraction settings
    extract_internal_confidence: bool = True
    extract_verbalized_certainty: bool = True

    # CQP settings
    cqp_template: str = "cell_attribution"  # "cell_attribution", "simple", "standard"
    cqp_max_tokens: int = 32
    cqp_temperature: float = 0.0

    # Alignment analysis settings
    alignment_threshold: float = 0.5  # Threshold for high/low confidence
    correctness_threshold: float = 0.5  # Cell F1 threshold for "correct"

    # Output settings
    save_per_sample_results: bool = True
    generate_alignment_report: bool = True


# Prompt template for cell attribution as multiple-choice
CELL_ATTRIBUTION_MCQ_PROMPT = """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

Below are all the cells in the table. Select the cell(s) that contain or directly support the answer.

CELLS:
{cell_options}

TASK: Identify the cell coordinate(s) that contain or support this answer. Return ONLY the cell coordinates (e.g., "A1, B2" or "=E7").

ATTRIBUTED CELLS:"""


class ConfidenceBenchmarkRunner:
    """
    Runner for confidence-probability alignment benchmark.

    Orchestrates the complete evaluation pipeline:
    1. Load dataset and model
    2. Run inference with confidence extraction
    3. Query verbalized certainty
    4. Compute alignment metrics
    5. Generate reports
    """

    def __init__(self, config: ConfidenceBenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config: Configuration for the benchmark
        """
        self.config = config
        self.dataset: Optional[VisualCiteDataset] = None
        self.model_manager: Optional[ModelManager] = None
        self.results: List[CellConfidenceResult] = []

        # Set up output directory
        self.output_dir = Path(config.output_dir) / "confidence_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the benchmark."""
        log_file = self.output_dir / f"confidence_benchmark_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def setup(self):
        """Load dataset and initialize model manager."""
        logger.info("Setting up confidence benchmark...")

        # Load dataset
        self.dataset = VisualCiteDataset(
            jsonl_path=self.config.jsonl_path,
            split=self.config.dataset_split,
            max_samples=self.config.max_samples
        )
        self.dataset.load()
        logger.info(f"Loaded {len(self.dataset)} samples")

        # Initialize model manager
        self.model_manager = ModelManager(self.config)

    def run_single_sample_confidence(
        self,
        sample: VisualCiteSample,
        model: Any,
        representation: DataRepresentation
    ) -> CellConfidenceResult:
        """
        Run confidence evaluation on a single sample.

        Args:
            sample: The sample to evaluate
            model: The model to use
            representation: Data representation format

        Returns:
            CellConfidenceResult with confidence data
        """
        # Get table representation
        table_repr = self.dataset.get_table_representation(
            sample, representation.value
        )

        # Get all cells and their values
        all_cells = get_all_table_cells(sample.table_json)
        cell_values = get_cell_values(sample.table_json)

        # Get ground truth cells
        ground_truth_cells = self.dataset.get_ground_truth_cells(sample)

        # Build prompt for cell attribution
        if isinstance(table_repr, Image.Image):
            # Image-based representation
            table_text = "[Image of table provided]"
            image = table_repr
        else:
            # Text-based representation
            table_text = table_repr if isinstance(table_repr, str) else json.dumps(table_repr)
            image = None

        # Create cell options string (show sample of cells)
        cell_options_list = []
        for cell in all_cells[:50]:  # Limit to 50 cells
            value = cell_values.get(cell, "")
            if value:
                cell_options_list.append(f"{cell}: {value[:50]}")
        cell_options = "\n".join(cell_options_list)
        if len(all_cells) > 50:
            cell_options += f"\n... ({len(all_cells)} total cells)"

        prompt = CELL_ATTRIBUTION_MCQ_PROMPT.format(
            table=table_text[:3000] if len(table_text) > 3000 else table_text,
            question=sample.question,
            answer=sample.answer,
            cell_options=cell_options
        )

        # Run inference
        start_time = time.perf_counter()

        if image is not None:
            result = model.generate_with_image(prompt, image)
        else:
            result = model.generate_text_only(prompt)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Parse predicted cells
        predicted_cells = parse_model_output(result.output_text)

        # Evaluate cell metrics
        cell_metrics = evaluate_single_prediction(predicted_cells, ground_truth_cells)

        # Initialize confidence result
        confidence_result = CellConfidenceResult(
            sample_id=sample.id,
            question=sample.question,
            answer=sample.answer,
            all_cells=all_cells,
            predicted_cells=predicted_cells,
            cell_confidences={},
            ground_truth_cells=ground_truth_cells,
            cell_precision=cell_metrics.cell_precision,
            cell_recall=cell_metrics.cell_recall,
            cell_f1=cell_metrics.cell_f1,
            exact_match=cell_metrics.exact_match,
            inference_time_ms=inference_time,
            model_name=model.model_name,
            representation=representation.value
        )

        # Extract verbalized certainty if enabled
        if self.config.extract_verbalized_certainty:
            try:
                certainty_extractor = VerbalizedCertaintyExtractor(
                    model=model.model,
                    processor=model.processor,
                    max_new_tokens=self.config.cqp_max_tokens,
                    temperature=self.config.cqp_temperature
                )

                # Build table string for CQP
                table_for_cqp = sample.table_md if sample.table_md else get_json_table_as_readable(sample.table_json)

                cqp_start = time.perf_counter()
                verbalized_result = certainty_extractor.query_confidence(
                    question=sample.question,
                    answer=sample.answer,
                    predicted_cells=predicted_cells,
                    table=table_for_cqp,
                    all_cells=all_cells,
                    image=image
                )
                cqp_time = (time.perf_counter() - cqp_start) * 1000

                confidence_result.verbalized_result = verbalized_result
                confidence_result.aggregate_verbalized_certainty = verbalized_result.certainty_score
                confidence_result.confidence_query_time_ms = cqp_time

                logger.debug(
                    f"Sample {sample.id}: Verbalized certainty = {verbalized_result.certainty_score:.2f} "
                    f"({verbalized_result.certainty_level.text if verbalized_result.certainty_level else 'N/A'})"
                )

            except Exception as e:
                logger.warning(f"Failed to extract verbalized certainty for {sample.id}: {e}")

        # For internal confidence, we use cell F1 as a proxy since we don't have
        # direct access to per-token logits in the current setup.
        # A more sophisticated approach would require modifying the model handler
        # to return logits during generation.
        confidence_result.aggregate_internal_confidence = cell_metrics.cell_f1

        return confidence_result

    def run_benchmark_configuration(
        self,
        model_name: str,
        representation: DataRepresentation
    ) -> List[CellConfidenceResult]:
        """
        Run benchmark for a specific model and representation configuration.

        Args:
            model_name: Model to use
            representation: Data representation format

        Returns:
            List of CellConfidenceResult for all samples
        """
        logger.info(f"Running confidence benchmark: {model_name} / {representation.value}")

        # Get model
        model = self.model_manager.get_model(model_name)

        results = []
        for i, sample in enumerate(self.dataset):
            try:
                result = self.run_single_sample_confidence(
                    sample, model, representation
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(self.dataset)} samples")

            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                continue

        return results

    def run_full_benchmark(self) -> Dict[str, AlignmentMetrics]:
        """
        Run the complete confidence benchmark across all configurations.

        Returns:
            Dictionary mapping configuration key to AlignmentMetrics
        """
        logger.info("Starting full confidence benchmark...")

        all_results = {}
        all_metrics = {}

        for model_name in self.config.models:
            for representation in self.config.representations:
                config_key = f"{model_name.split('/')[-1]}_{representation.value}"

                try:
                    results = self.run_benchmark_configuration(model_name, representation)
                    all_results[config_key] = results

                    # Compute alignment metrics
                    alignment_metrics = compute_cell_alignment_metrics(results)
                    all_metrics[config_key] = alignment_metrics

                    # Log summary
                    logger.info(
                        f"{config_key}: Spearman ρ = {alignment_metrics.spearman_rho:.4f} "
                        f"(p = {alignment_metrics.spearman_p_value:.4f})"
                    )

                except Exception as e:
                    logger.error(f"Error in configuration {config_key}: {e}")
                    continue

        # Save results
        self._save_results(all_results, all_metrics)

        return all_metrics

    def _save_results(
        self,
        all_results: Dict[str, List[CellConfidenceResult]],
        all_metrics: Dict[str, AlignmentMetrics]
    ):
        """Save all results and metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save per-sample results
        if self.config.save_per_sample_results:
            for config_key, results in all_results.items():
                results_file = self.output_dir / f"{config_key}_results_{timestamp}.jsonl"
                with open(results_file, 'w') as f:
                    for r in results:
                        record = {
                            "sample_id": r.sample_id,
                            "question": r.question,
                            "answer": r.answer,
                            "predicted_cells": r.predicted_cells,
                            "ground_truth_cells": r.ground_truth_cells,
                            "cell_precision": r.cell_precision,
                            "cell_recall": r.cell_recall,
                            "cell_f1": r.cell_f1,
                            "exact_match": r.exact_match,
                            "internal_confidence": r.aggregate_internal_confidence,
                            "verbalized_certainty": r.aggregate_verbalized_certainty,
                            "verbalized_response": r.verbalized_result.raw_response if r.verbalized_result else None,
                            "inference_time_ms": r.inference_time_ms,
                            "confidence_query_time_ms": r.confidence_query_time_ms,
                        }
                        f.write(json.dumps(record) + "\n")
                logger.info(f"Saved results to {results_file}")

        # Save alignment metrics summary
        metrics_summary = {}
        for config_key, metrics in all_metrics.items():
            metrics_summary[config_key] = alignment_metrics_to_dict(metrics)

        summary_file = self.output_dir / f"alignment_metrics_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Saved metrics summary to {summary_file}")

        # Generate alignment report
        if self.config.generate_alignment_report:
            report_file = self.output_dir / f"alignment_report_{timestamp}.md"
            with open(report_file, 'w') as f:
                f.write("# Confidence-Probability Alignment Report\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")

                for config_key, metrics in all_metrics.items():
                    f.write(f"## Configuration: {config_key}\n\n")
                    f.write("```\n")
                    f.write(format_alignment_report(metrics))
                    f.write("\n```\n\n")

            logger.info(f"Saved alignment report to {report_file}")

    def cleanup(self):
        """Clean up resources."""
        if self.model_manager:
            self.model_manager.cleanup()


def run_confidence_benchmark(
    config: Optional[ConfidenceBenchmarkConfig] = None,
    **kwargs
) -> Dict[str, AlignmentMetrics]:
    """
    Convenience function to run the confidence benchmark.

    Args:
        config: Optional configuration object
        **kwargs: Additional configuration parameters

    Returns:
        Dictionary mapping configuration key to AlignmentMetrics
    """
    if config is None:
        config = ConfidenceBenchmarkConfig(**kwargs)

    runner = ConfidenceBenchmarkRunner(config)
    try:
        runner.setup()
        results = runner.run_full_benchmark()
        return results
    finally:
        runner.cleanup()


if __name__ == "__main__":
    # Example usage
    config = ConfidenceBenchmarkConfig(
        jsonl_path="../visualcite.jsonl",
        dataset_split="dev",
        max_samples=50,  # Small sample for testing
        models=["Qwen/Qwen3-VL-2B-Instruct"],
        representations=[DataRepresentation.MARKDOWN],
        output_dir="benchmark_results",
        extract_internal_confidence=True,
        extract_verbalized_certainty=True,
    )

    results = run_confidence_benchmark(config)

    for config_key, metrics in results.items():
        print(f"\n{config_key}:")
        print(format_alignment_report(metrics))
