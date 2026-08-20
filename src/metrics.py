"""
Evaluation metrics for cell attribution benchmark
"""
from typing import List, Set, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CellMetrics:
    """Metrics for a single prediction"""
    # Raw data
    predicted_cells: List[str]
    ground_truth_cells: List[str]
    
    # Cell-level metrics
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    
    # Row-level metrics
    row_precision: float = 0.0
    row_recall: float = 0.0
    row_f1: float = 0.0
    
    # Column-level metrics
    col_precision: float = 0.0
    col_recall: float = 0.0
    col_f1: float = 0.0
    
    # Exact match
    exact_match: bool = False
    
    # Partial match (at least one correct cell)
    partial_match: bool = False


def extract_row_col(cell: str) -> Tuple[int, str]:
    """
    Extract row number and column letters from cell coordinate.
    E.g., "B3" -> (3, "B"), "AA15" -> (15, "AA")
    """
    import re
    match = re.match(r'^([A-Za-z]+)(\d+)$', cell.strip())
    if not match:
        return None, None
    col_str, row_str = match.groups()
    return int(row_str), col_str.upper()


def compute_precision_recall_f1(
    predicted: Set[Any],
    ground_truth: Set[Any]
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score"""
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    
    if not predicted:
        return 0.0, 0.0, 0.0
    
    if not ground_truth:
        return 0.0, 0.0, 0.0
    
    true_positives = len(predicted & ground_truth)
    precision = true_positives / len(predicted)
    recall = true_positives / len(ground_truth)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def evaluate_single_prediction(
    predicted_cells: List[str],
    ground_truth_cells: List[str]
) -> CellMetrics:
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        predicted_cells: List of predicted cell coordinates (e.g., ["A1", "B2"])
        ground_truth_cells: List of ground truth cell coordinates
    
    Returns:
        CellMetrics object with all computed metrics
    """
    metrics = CellMetrics(
        predicted_cells=predicted_cells,
        ground_truth_cells=ground_truth_cells
    )
    
    # Normalize to uppercase sets
    pred_set = set(c.upper() for c in predicted_cells)
    gt_set = set(c.upper() for c in ground_truth_cells)
    
    # Cell-level metrics
    metrics.cell_precision, metrics.cell_recall, metrics.cell_f1 = \
        compute_precision_recall_f1(pred_set, gt_set)
    
    # Extract rows and columns
    pred_rows = set()
    pred_cols = set()
    for cell in pred_set:
        row, col = extract_row_col(cell)
        if row is not None:
            pred_rows.add(row)
            pred_cols.add(col)
    
    gt_rows = set()
    gt_cols = set()
    for cell in gt_set:
        row, col = extract_row_col(cell)
        if row is not None:
            gt_rows.add(row)
            gt_cols.add(col)
    
    # Row-level metrics
    metrics.row_precision, metrics.row_recall, metrics.row_f1 = \
        compute_precision_recall_f1(pred_rows, gt_rows)
    
    # Column-level metrics
    metrics.col_precision, metrics.col_recall, metrics.col_f1 = \
        compute_precision_recall_f1(pred_cols, gt_cols)
    
    # Exact match (all predictions correct and complete)
    metrics.exact_match = pred_set == gt_set
    
    # Partial match (at least one correct cell)
    metrics.partial_match = len(pred_set & gt_set) > 0
    
    return metrics


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple predictions"""
    # Counts
    total_samples: int = 0
    valid_predictions: int = 0
    
    # Cell-level aggregates
    mean_cell_precision: float = 0.0
    mean_cell_recall: float = 0.0
    mean_cell_f1: float = 0.0
    
    # Row-level aggregates
    mean_row_precision: float = 0.0
    mean_row_recall: float = 0.0
    mean_row_f1: float = 0.0
    
    # Column-level aggregates
    mean_col_precision: float = 0.0
    mean_col_recall: float = 0.0
    mean_col_f1: float = 0.0
    
    # Match rates
    exact_match_rate: float = 0.0
    partial_match_rate: float = 0.0
    
    # Additional statistics
    avg_predicted_cells: float = 0.0
    avg_ground_truth_cells: float = 0.0
    
    # Per-sample metrics for detailed analysis
    sample_metrics: List[Dict[str, Any]] = field(default_factory=list)


def aggregate_metrics(metrics_list: List[CellMetrics]) -> AggregatedMetrics:
    """
    Aggregate metrics from multiple predictions.
    
    Args:
        metrics_list: List of CellMetrics from individual predictions
    
    Returns:
        AggregatedMetrics with means and totals
    """
    if not metrics_list:
        return AggregatedMetrics()
    
    agg = AggregatedMetrics()
    agg.total_samples = len(metrics_list)
    
    # Sum up all metrics
    cell_precision_sum = 0.0
    cell_recall_sum = 0.0
    cell_f1_sum = 0.0
    
    row_precision_sum = 0.0
    row_recall_sum = 0.0
    row_f1_sum = 0.0
    
    col_precision_sum = 0.0
    col_recall_sum = 0.0
    col_f1_sum = 0.0
    
    exact_match_count = 0
    partial_match_count = 0
    
    total_predicted = 0
    total_ground_truth = 0
    
    for m in metrics_list:
        cell_precision_sum += m.cell_precision
        cell_recall_sum += m.cell_recall
        cell_f1_sum += m.cell_f1
        
        row_precision_sum += m.row_precision
        row_recall_sum += m.row_recall
        row_f1_sum += m.row_f1
        
        col_precision_sum += m.col_precision
        col_recall_sum += m.col_recall
        col_f1_sum += m.col_f1
        
        if m.exact_match:
            exact_match_count += 1
        if m.partial_match:
            partial_match_count += 1
        
        total_predicted += len(m.predicted_cells)
        total_ground_truth += len(m.ground_truth_cells)
        
        # Store per-sample metrics
        agg.sample_metrics.append({
            'predicted_cells': m.predicted_cells,
            'ground_truth_cells': m.ground_truth_cells,
            'cell_precision': m.cell_precision,
            'cell_recall': m.cell_recall,
            'cell_f1': m.cell_f1,
            'exact_match': m.exact_match,
            'partial_match': m.partial_match
        })
    
    n = agg.total_samples
    agg.valid_predictions = n
    
    # Compute means
    agg.mean_cell_precision = cell_precision_sum / n
    agg.mean_cell_recall = cell_recall_sum / n
    agg.mean_cell_f1 = cell_f1_sum / n
    
    agg.mean_row_precision = row_precision_sum / n
    agg.mean_row_recall = row_recall_sum / n
    agg.mean_row_f1 = row_f1_sum / n
    
    agg.mean_col_precision = col_precision_sum / n
    agg.mean_col_recall = col_recall_sum / n
    agg.mean_col_f1 = col_f1_sum / n
    
    agg.exact_match_rate = exact_match_count / n
    agg.partial_match_rate = partial_match_count / n
    
    agg.avg_predicted_cells = total_predicted / n
    agg.avg_ground_truth_cells = total_ground_truth / n
    
    return agg


def metrics_to_dict(agg: AggregatedMetrics) -> Dict[str, Any]:
    """Convert AggregatedMetrics to dictionary for JSON serialization"""
    return {
        'total_samples': agg.total_samples,
        'valid_predictions': agg.valid_predictions,
        'cell_level': {
            'precision': round(agg.mean_cell_precision, 4),
            'recall': round(agg.mean_cell_recall, 4),
            'f1': round(agg.mean_cell_f1, 4)
        },
        'row_level': {
            'precision': round(agg.mean_row_precision, 4),
            'recall': round(agg.mean_row_recall, 4),
            'f1': round(agg.mean_row_f1, 4)
        },
        'column_level': {
            'precision': round(agg.mean_col_precision, 4),
            'recall': round(agg.mean_col_recall, 4),
            'f1': round(agg.mean_col_f1, 4)
        },
        'exact_match_rate': round(agg.exact_match_rate, 4),
        'partial_match_rate': round(agg.partial_match_rate, 4),
        'avg_predicted_cells': round(agg.avg_predicted_cells, 2),
        'avg_ground_truth_cells': round(agg.avg_ground_truth_cells, 2)
    }


def format_metrics_table(agg: AggregatedMetrics) -> str:
    """Format metrics as a readable table"""
    lines = [
        "=" * 60,
        f"EVALUATION RESULTS ({agg.total_samples} samples)",
        "=" * 60,
        "",
        "CELL-LEVEL METRICS:",
        f"  Precision: {agg.mean_cell_precision:.4f}",
        f"  Recall:    {agg.mean_cell_recall:.4f}",
        f"  F1 Score:  {agg.mean_cell_f1:.4f}",
        "",
        "ROW-LEVEL METRICS:",
        f"  Precision: {agg.mean_row_precision:.4f}",
        f"  Recall:    {agg.mean_row_recall:.4f}",
        f"  F1 Score:  {agg.mean_row_f1:.4f}",
        "",
        "COLUMN-LEVEL METRICS:",
        f"  Precision: {agg.mean_col_precision:.4f}",
        f"  Recall:    {agg.mean_col_recall:.4f}",
        f"  F1 Score:  {agg.mean_col_f1:.4f}",
        "",
        "MATCH RATES:",
        f"  Exact Match:   {agg.exact_match_rate:.4f} ({int(agg.exact_match_rate * agg.total_samples)}/{agg.total_samples})",
        f"  Partial Match: {agg.partial_match_rate:.4f} ({int(agg.partial_match_rate * agg.total_samples)}/{agg.total_samples})",
        "",
        "PREDICTION STATISTICS:",
        f"  Avg Predicted Cells:    {agg.avg_predicted_cells:.2f}",
        f"  Avg Ground Truth Cells: {agg.avg_ground_truth_cells:.2f}",
        "=" * 60
    ]
    return "\n".join(lines)