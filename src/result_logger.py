"""
Result logging and export for benchmarks
"""
import os
import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from metrics import AggregatedMetrics, metrics_to_dict, format_metrics_table

logger = logging.getLogger(__name__)


@dataclass
class InstanceLog:
    """Log entry for a single inference instance"""
    sample_id: str
    model_name: str
    representation: str
    strategy: str
    question: str
    answer: str
    ground_truth_cells: List[str]
    predicted_cells: List[str]
    raw_output: str
    input_prompt: str
    inference_time_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cell_precision: float
    cell_recall: float
    cell_f1: float
    exact_match: bool
    partial_match: bool
    timestamp: str
    internal_confidence: Optional[float] = None
    verbalized_certainty: Optional[float] = None
    cell_confidences: Optional[Dict[str, float]] = None


@dataclass
class VCInstanceLog:
    """Log entry for a verbalized certainty query instance"""
    sample_id: str
    model_name: str
    representation: str
    strategy: str
    question: str
    answer: str
    predicted_cells: List[str]
    vc_prompt: str
    vc_response: str
    certainty_level: Optional[str]
    certainty_score: float
    parse_success: bool
    query_time_ms: float
    timestamp: str


class ResultLogger:
    """Manages logging of benchmark results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = os.path.join(output_dir, "instance_logs")
        self.vc_logs_dir = os.path.join(output_dir, "verbalized_certainty_logs")
        self.summaries_dir = os.path.join(output_dir, "summaries")
        self.exports_dir = os.path.join(output_dir, "exports")
        self.uq_dir = os.path.join(output_dir, "uncertainty")
        
        for d in [self.logs_dir, self.vc_logs_dir, self.summaries_dir, self.exports_dir, self.uq_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_instance(self, log_entry: InstanceLog) -> None:
        """Log a single instance result"""
        # Create filename based on configuration
        filename = f"{log_entry.model_name.split('/')[-1]}_{log_entry.representation}_{log_entry.strategy}.jsonl"
        filepath = os.path.join(self.logs_dir, filename)
        
        # Append to JSONL file
        with open(filepath, 'a') as f:
            f.write(json.dumps(asdict(log_entry)) + '\n')
    
    def log_vc_instance(self, log_entry: VCInstanceLog) -> None:
        """Log a single verbalized certainty query instance"""
        # Create filename based on configuration
        filename = f"{log_entry.model_name.split('/')[-1]}_{log_entry.representation}_{log_entry.strategy}_vc.jsonl"
        filepath = os.path.join(self.vc_logs_dir, filename)
        
        # Append to JSONL file
        with open(filepath, 'a') as f:
            f.write(json.dumps(asdict(log_entry)) + '\n')
    
    def save_aggregated_results(
        self,
        model_name: str,
        representation: str,
        strategy: str,
        metrics: AggregatedMetrics,
        total_time_seconds: float,
        avg_inference_time_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save aggregated metrics for a benchmark run"""
        if avg_inference_time_ms is None:
            # Backward-compatible fallback: older code paths may not provide this.
            # Note: AggregatedMetrics.sample_metrics does not necessarily include timing fields.
            avg_inference_time_ms = (
                sum(m.get('inference_time_ms', 0) for m in metrics.sample_metrics) /
                len(metrics.sample_metrics)
            ) if metrics.sample_metrics else 0.0

        result = {
            'run_id': self.run_id,
            'model_name': model_name,
            'representation': representation,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': round(total_time_seconds, 2),
            'avg_inference_time_ms': round(float(avg_inference_time_ms), 2),
            'metrics': metrics_to_dict(metrics)
        }

        if extra:
            result.update(extra)
        
        # Save individual summary
        filename = f"{model_name.split('/')[-1]}_{representation}_{strategy}_summary.json"
        filepath = os.path.join(self.summaries_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved summary: {filepath}")
        return filepath

    def save_uncertainty_results(
        self,
        model_name: str,
        representation: str,
        strategy: str,
        uq_summary: Dict[str, Any]
    ) -> str:
        """Save conformal UQ summaries for a single configuration."""
        filename = f"{model_name.split('/')[-1]}_{representation}_{strategy}_conformal_uq_{self.run_id}.json"
        filepath = os.path.join(self.uq_dir, filename)
        payload = {
            'run_id': self.run_id,
            'model_name': model_name,
            'representation': representation,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'conformal_uq': uq_summary,
        }
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved conformal UQ summary: {filepath}")
        return filepath
    
    def export_all_results_csv(self, results: List[Dict[str, Any]]) -> str:
        """Export all results to CSV for easy analysis"""
        if not results:
            return ""
        
        filepath = os.path.join(self.exports_dir, f"benchmark_results_{self.run_id}.csv")
        
        # Flatten nested metrics
        flat_results = []
        for r in results:
            flat = {
                'run_id': r.get('run_id', ''),
                'model_name': r.get('model_name', ''),
                'representation': r.get('representation', ''),
                'strategy': r.get('strategy', ''),
                'total_samples': r.get('metrics', {}).get('total_samples', 0),
                'total_time_seconds': r.get('total_time_seconds', 0),
                'avg_inference_time_ms': r.get('avg_inference_time_ms', 0),
                'cell_precision': r.get('metrics', {}).get('cell_level', {}).get('precision', 0),
                'cell_recall': r.get('metrics', {}).get('cell_level', {}).get('recall', 0),
                'cell_f1': r.get('metrics', {}).get('cell_level', {}).get('f1', 0),
                'row_precision': r.get('metrics', {}).get('row_level', {}).get('precision', 0),
                'row_recall': r.get('metrics', {}).get('row_level', {}).get('recall', 0),
                'row_f1': r.get('metrics', {}).get('row_level', {}).get('f1', 0),
                'col_precision': r.get('metrics', {}).get('column_level', {}).get('precision', 0),
                'col_recall': r.get('metrics', {}).get('column_level', {}).get('recall', 0),
                'col_f1': r.get('metrics', {}).get('column_level', {}).get('f1', 0),
                'exact_match_rate': r.get('metrics', {}).get('exact_match_rate', 0),
                'partial_match_rate': r.get('metrics', {}).get('partial_match_rate', 0),
            }
            flat_results.append(flat)
        
        # Write CSV
        fieldnames = list(flat_results[0].keys())
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)
        
        logger.info(f"Exported results to CSV: {filepath}")
        return filepath
    
    def export_detailed_instances_csv(self) -> str:
        """Export all instance logs to a single CSV"""
        filepath = os.path.join(self.exports_dir, f"detailed_instances_{self.run_id}.csv")
        
        all_instances = []
        
        # Read all JSONL files
        for filename in os.listdir(self.logs_dir):
            if filename.endswith('.jsonl'):
                with open(os.path.join(self.logs_dir, filename), 'r') as f:
                    for line in f:
                        if line.strip():
                            instance = json.loads(line)
                            # Flatten lists for CSV
                            instance['ground_truth_cells'] = ','.join(instance.get('ground_truth_cells', []))
                            instance['predicted_cells'] = ','.join(instance.get('predicted_cells', []))
                            all_instances.append(instance)
        
        if not all_instances:
            return ""
        
        # Write CSV
        fieldnames = [
            'sample_id', 'model_name', 'representation', 'strategy',
            'question', 'answer', 'ground_truth_cells', 'predicted_cells',
            'cell_precision', 'cell_recall', 'cell_f1',
            'exact_match', 'partial_match',
            'inference_time_ms', 'input_tokens', 'output_tokens',
            'timestamp'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_instances)
        
        logger.info(f"Exported detailed instances to CSV: {filepath}")
        return filepath
    
    def export_vc_instances_csv(self) -> str:
        """Export all verbalized certainty logs to a single CSV"""
        filepath = os.path.join(self.exports_dir, f"verbalized_certainty_{self.run_id}.csv")
        
        all_instances = []
        
        # Read all VC JSONL files
        if os.path.exists(self.vc_logs_dir):
            for filename in os.listdir(self.vc_logs_dir):
                if filename.endswith('.jsonl'):
                    with open(os.path.join(self.vc_logs_dir, filename), 'r') as f:
                        for line in f:
                            if line.strip():
                                instance = json.loads(line)
                                # Flatten lists for CSV
                                instance['predicted_cells'] = ','.join(instance.get('predicted_cells', []))
                                # Truncate long prompts/responses for CSV readability
                                if 'vc_prompt' in instance and len(instance['vc_prompt']) > 500:
                                    instance['vc_prompt_preview'] = instance['vc_prompt'][:500] + '...'
                                else:
                                    instance['vc_prompt_preview'] = instance.get('vc_prompt', '')
                                all_instances.append(instance)
        
        if not all_instances:
            logger.info("No verbalized certainty logs to export")
            return ""
        
        # Write CSV
        fieldnames = [
            'sample_id', 'model_name', 'representation', 'strategy',
            'question', 'answer', 'predicted_cells',
            'vc_response', 'certainty_level', 'certainty_score',
            'parse_success', 'query_time_ms', 'timestamp'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_instances)
        
        logger.info(f"Exported verbalized certainty instances to CSV: {filepath}")
        return filepath
    
    def generate_report(self, all_results: List[Dict[str, Any]]) -> str:
        """Generate a markdown report of all results"""
        report_lines = [
            "# VisualCite Attribution Benchmark Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\nRun ID: {self.run_id}",
            "\n## Summary",
            f"\nTotal configurations tested: {len(all_results)}",
        ]
        
        # Group by model
        by_model = {}
        for r in all_results:
            model = r['model_name']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
        
        report_lines.append(f"\nModels tested: {len(by_model)}")
        
        # Create results table
        report_lines.extend([
            "\n## Results Table",
            "\n| Model | Representation | Strategy | Cell F1 | Row F1 | Col F1 | Exact Match | Time (s) |",
            "|-------|----------------|----------|---------|--------|--------|-------------|----------|"
        ])
        
        for r in sorted(all_results, key=lambda x: (x['model_name'], x['representation'], x['strategy'])):
            model_short = r['model_name'].split('/')[-1]
            metrics = r['metrics']
            row = f"| {model_short} | {r['representation']} | {r['strategy']} | " \
                  f"{metrics['cell_level']['f1']:.4f} | " \
                  f"{metrics['row_level']['f1']:.4f} | " \
                  f"{metrics['column_level']['f1']:.4f} | " \
                  f"{metrics['exact_match_rate']:.4f} | " \
                  f"{r['total_time_seconds']:.1f} |"
            report_lines.append(row)
        
        # Best configurations
        report_lines.extend([
            "\n## Best Configurations",
            "\n### By Cell F1 Score"
        ])
        
        sorted_by_f1 = sorted(all_results, key=lambda x: x['metrics']['cell_level']['f1'], reverse=True)[:5]
        for i, r in enumerate(sorted_by_f1, 1):
            report_lines.append(
                f"{i}. {r['model_name'].split('/')[-1]} + {r['representation']} + {r['strategy']}: "
                f"F1={r['metrics']['cell_level']['f1']:.4f}"
            )
        
        report_lines.extend([
            "\n### By Exact Match Rate"
        ])
        
        sorted_by_em = sorted(all_results, key=lambda x: x['metrics']['exact_match_rate'], reverse=True)[:5]
        for i, r in enumerate(sorted_by_em, 1):
            report_lines.append(
                f"{i}. {r['model_name'].split('/')[-1]} + {r['representation']} + {r['strategy']}: "
                f"EM={r['metrics']['exact_match_rate']:.4f}"
            )
        
        # Write report
        report_path = os.path.join(self.output_dir, f"benchmark_report_{self.run_id}.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated report: {report_path}")
        return report_path


def create_instance_log(
    sample_id: str,
    model_name: str,
    representation: str,
    strategy: str,
    question: str,
    answer: str,
    ground_truth_cells: List[str],
    predicted_cells: List[str],
    raw_output: str,
    input_prompt: str,
    inference_time_ms: float,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cell_metrics: Dict[str, float],
    confidence_data: Optional[Dict[str, Any]] = None
) -> InstanceLog:
    """Factory function to create an InstanceLog"""
    if confidence_data is None:
        confidence_data = {}
    
    return InstanceLog(
        sample_id=sample_id,
        model_name=model_name,
        representation=representation,
        strategy=strategy,
        question=question,
        answer=answer,
        ground_truth_cells=ground_truth_cells,
        predicted_cells=predicted_cells,
        raw_output=raw_output,
        input_prompt=input_prompt,
        inference_time_ms=inference_time_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cell_precision=cell_metrics.get('precision', 0.0),
        cell_recall=cell_metrics.get('recall', 0.0),
        cell_f1=cell_metrics.get('f1', 0.0),
        exact_match=cell_metrics.get('exact_match', False),
        partial_match=cell_metrics.get('partial_match', False),
        timestamp=datetime.now().isoformat(),
        internal_confidence=confidence_data.get('internal_confidence'),
        verbalized_certainty=confidence_data.get('verbalized_certainty'),
        cell_confidences=confidence_data.get('cell_confidences')
    )

def create_vc_instance_log(
    sample_id: str,
    model_name: str,
    representation: str,
    strategy: str,
    question: str,
    answer: str,
    predicted_cells: List[str],
    verbalized_result: Any,
    query_time_ms: float
) -> VCInstanceLog:
    """Factory function to create a VCInstanceLog"""
    return VCInstanceLog(
        sample_id=sample_id,
        model_name=model_name,
        representation=representation,
        strategy=strategy,
        question=question,
        answer=answer,
        predicted_cells=predicted_cells,
        vc_prompt=verbalized_result.query_prompt,
        vc_response=verbalized_result.raw_response,
        certainty_level=verbalized_result.certainty_level.text if verbalized_result.certainty_level else None,
        certainty_score=verbalized_result.certainty_score,
        parse_success=verbalized_result.parse_success,
        query_time_ms=query_time_ms,
        timestamp=datetime.now().isoformat()
    )