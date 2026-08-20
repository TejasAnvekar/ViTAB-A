"""
Alignment Metrics Computation for Confidence-Probability Alignment.

Implements the alignment evaluation methodology from the paper
"Confidence Under the Hood" (ACL 2024), Section 2.4.

Primary metric: Spearman's rank correlation coefficient (ρ)
- Non-parametric test measuring association between two variables
- Ideal for non-normally distributed confidence values
- Does not require normal distribution assumption

Additional metrics included:
- Pearson correlation coefficient
- Kendall's tau
- Distribution statistics
- Alignment type classification counts
"""
import logging
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

try:
    # Package import (e.g., `python -m src.benchmark_runner`)
    from .confidence_types import (
        ConfidenceResult,
        AlignmentMetrics,
        AlignmentType,
        CellConfidenceResult,
        classify_alignment,
    )
except ImportError:  # pragma: no cover
    # Script import (e.g., `cd src; python benchmark_runner.py`)
    from confidence_types import (
        ConfidenceResult,
        AlignmentMetrics,
        AlignmentType,
        CellConfidenceResult,
        classify_alignment,
    )

logger = logging.getLogger(__name__)


def compute_spearman_correlation(
    internal_confidences: List[float],
    verbalized_certainties: List[float]
) -> Tuple[float, float]:
    """
    Compute Spearman's rank correlation coefficient.

    From paper Equation 3:
    ρ = 1 - (6 * Σd²) / (n * (n² - 1))

    where d_i is the difference between ranks of corresponding values.

    Args:
        internal_confidences: List of internal confidence scores
        verbalized_certainties: List of verbalized certainty scores

    Returns:
        Tuple of (rho, p_value)
    """
    if len(internal_confidences) != len(verbalized_certainties):
        raise ValueError("Lists must have same length")

    if len(internal_confidences) < 3:
        logger.warning("Too few samples for meaningful correlation")
        return 0.0, 1.0

    # Use scipy for robust computation
    rho, p_value = stats.spearmanr(internal_confidences, verbalized_certainties)

    # Handle NaN cases
    if math.isnan(rho):
        logger.warning("Spearman correlation returned NaN")
        return 0.0, 1.0

    return float(rho), float(p_value)


def compute_pearson_correlation(
    internal_confidences: List[float],
    verbalized_certainties: List[float]
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Included for comparison with Spearman, though the paper primarily
    uses Spearman due to non-normal distributions.

    Args:
        internal_confidences: List of internal confidence scores
        verbalized_certainties: List of verbalized certainty scores

    Returns:
        Tuple of (r, p_value)
    """
    if len(internal_confidences) != len(verbalized_certainties):
        raise ValueError("Lists must have same length")

    if len(internal_confidences) < 3:
        return 0.0, 1.0

    r, p_value = stats.pearsonr(internal_confidences, verbalized_certainties)

    if math.isnan(r):
        return 0.0, 1.0

    return float(r), float(p_value)


def compute_kendall_tau(
    internal_confidences: List[float],
    verbalized_certainties: List[float]
) -> Tuple[float, float]:
    """
    Compute Kendall's tau correlation coefficient.

    Another rank-based correlation measure, more robust to ties.

    Args:
        internal_confidences: List of internal confidence scores
        verbalized_certainties: List of verbalized certainty scores

    Returns:
        Tuple of (tau, p_value)
    """
    if len(internal_confidences) != len(verbalized_certainties):
        raise ValueError("Lists must have same length")

    if len(internal_confidences) < 3:
        return 0.0, 1.0

    tau, p_value = stats.kendalltau(internal_confidences, verbalized_certainties)

    if math.isnan(tau):
        return 0.0, 1.0

    return float(tau), float(p_value)


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute distribution statistics for a list of values.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with mean, std, median, min, max
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0
        }

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def compute_alignment_metrics(
    results: List[ConfidenceResult],
    alignment_threshold: float = 0.5
) -> AlignmentMetrics:
    """
    Compute comprehensive alignment metrics from a list of confidence results.

    This is the main function for evaluating confidence-probability alignment
    following the methodology in the paper.

    Args:
        results: List of ConfidenceResult objects with both confidence measures
        alignment_threshold: Threshold for high/low confidence classification

    Returns:
        AlignmentMetrics with all computed statistics
    """
    metrics = AlignmentMetrics()
    metrics.n_samples = len(results)

    # Filter to valid samples (those with both confidence measures)
    valid_results = [
        r for r in results
        if r.internal_confidence is not None and r.verbalized_certainty is not None
    ]
    metrics.n_valid_samples = len(valid_results)

    if not valid_results:
        logger.warning("No valid samples with both confidence measures")
        return metrics

    # Extract confidence values
    internal_values = [r.internal_confidence.adjusted_confidence for r in valid_results]
    verbalized_values = [r.verbalized_certainty.certainty_score for r in valid_results]

    # Compute correlation metrics
    metrics.spearman_rho, metrics.spearman_p_value = compute_spearman_correlation(
        internal_values, verbalized_values
    )

    metrics.pearson_r, metrics.pearson_p_value = compute_pearson_correlation(
        internal_values, verbalized_values
    )

    metrics.kendall_tau, metrics.kendall_p_value = compute_kendall_tau(
        internal_values, verbalized_values
    )

    # Compute distribution statistics
    internal_stats = compute_distribution_stats(internal_values)
    metrics.internal_mean = internal_stats["mean"]
    metrics.internal_std = internal_stats["std"]
    metrics.internal_median = internal_stats["median"]
    metrics.internal_min = internal_stats["min"]
    metrics.internal_max = internal_stats["max"]

    verbalized_stats = compute_distribution_stats(verbalized_values)
    metrics.verbalized_mean = verbalized_stats["mean"]
    metrics.verbalized_std = verbalized_stats["std"]
    metrics.verbalized_median = verbalized_stats["median"]
    metrics.verbalized_min = verbalized_stats["min"]
    metrics.verbalized_max = verbalized_stats["max"]

    # Classify alignment types
    for r in valid_results:
        internal = r.internal_confidence.adjusted_confidence
        verbalized = r.verbalized_certainty.certainty_score

        alignment_type = classify_alignment(internal, verbalized)
        r.alignment_type = alignment_type

        if alignment_type == AlignmentType.CONSISTENT_ALIGNMENT:
            metrics.consistent_alignment_count += 1
        elif alignment_type == AlignmentType.INTERNAL_OVERCONFIDENCE:
            metrics.internal_overconfidence_count += 1
        elif alignment_type == AlignmentType.EXTERNAL_OVERCONFIDENCE:
            metrics.external_overconfidence_count += 1
        elif alignment_type == AlignmentType.CONSISTENT_DISCORDANCE:
            metrics.consistent_discordance_count += 1

    # Compute correctness-confidence relationship
    correct_results = [r for r in valid_results if r.is_correct is True]
    incorrect_results = [r for r in valid_results if r.is_correct is False]

    metrics.correct_samples_count = len(correct_results)
    metrics.incorrect_samples_count = len(incorrect_results)

    if correct_results:
        metrics.mean_confidence_when_correct = np.mean([
            r.internal_confidence.adjusted_confidence for r in correct_results
        ])
        metrics.mean_verbalized_when_correct = np.mean([
            r.verbalized_certainty.certainty_score for r in correct_results
        ])

    if incorrect_results:
        metrics.mean_confidence_when_incorrect = np.mean([
            r.internal_confidence.adjusted_confidence for r in incorrect_results
        ])
        metrics.mean_verbalized_when_incorrect = np.mean([
            r.verbalized_certainty.certainty_score for r in incorrect_results
        ])

    # Store sample results
    metrics.sample_results = valid_results

    return metrics


def compute_cell_alignment_metrics(
    results: List[CellConfidenceResult]
) -> AlignmentMetrics:
    """
    Compute alignment metrics specifically for cell attribution results.

    Args:
        results: List of CellConfidenceResult objects

    Returns:
        AlignmentMetrics with computed statistics
    """
    metrics = AlignmentMetrics()
    metrics.n_samples = len(results)

    # Filter valid results: require both confidence measures.
    # Note: when resuming from checkpoints we may not have the full
    # VerbalizedCertaintyResult object, but we should still be able to compute
    # alignment from the stored aggregate certainty score.
    valid_results = [
        r for r in results
        if r.aggregate_internal_confidence is not None
        and (
            r.verbalized_result is not None
            or r.aggregate_verbalized_certainty is not None
        )
    ]
    metrics.n_valid_samples = len(valid_results)

    if not valid_results:
        logger.warning("No valid cell confidence results")
        return metrics

    # Extract values
    internal_values = [float(r.aggregate_internal_confidence) for r in valid_results]
    verbalized_values: List[float] = []
    for r in valid_results:
        if r.verbalized_result is not None:
            verbalized_values.append(float(r.verbalized_result.certainty_score))
        else:
            # Resumed-run path
            verbalized_values.append(float(r.aggregate_verbalized_certainty))

    # Compute correlations
    metrics.spearman_rho, metrics.spearman_p_value = compute_spearman_correlation(
        internal_values, verbalized_values
    )

    metrics.pearson_r, metrics.pearson_p_value = compute_pearson_correlation(
        internal_values, verbalized_values
    )

    metrics.kendall_tau, metrics.kendall_p_value = compute_kendall_tau(
        internal_values, verbalized_values
    )

    # Distribution stats
    internal_stats = compute_distribution_stats(internal_values)
    metrics.internal_mean = internal_stats["mean"]
    metrics.internal_std = internal_stats["std"]
    metrics.internal_median = internal_stats["median"]
    metrics.internal_min = internal_stats["min"]
    metrics.internal_max = internal_stats["max"]

    verbalized_stats = compute_distribution_stats(verbalized_values)
    metrics.verbalized_mean = verbalized_stats["mean"]
    metrics.verbalized_std = verbalized_stats["std"]
    metrics.verbalized_median = verbalized_stats["median"]
    metrics.verbalized_min = verbalized_stats["min"]
    metrics.verbalized_max = verbalized_stats["max"]

    # Alignment types
    for r in valid_results:
        verbalized_score = (
            float(r.verbalized_result.certainty_score)
            if r.verbalized_result is not None
            else float(r.aggregate_verbalized_certainty)
        )
        alignment_type = classify_alignment(
            float(r.aggregate_internal_confidence),
            verbalized_score
        )

        if alignment_type == AlignmentType.CONSISTENT_ALIGNMENT:
            metrics.consistent_alignment_count += 1
        elif alignment_type == AlignmentType.INTERNAL_OVERCONFIDENCE:
            metrics.internal_overconfidence_count += 1
        elif alignment_type == AlignmentType.EXTERNAL_OVERCONFIDENCE:
            metrics.external_overconfidence_count += 1
        elif alignment_type == AlignmentType.CONSISTENT_DISCORDANCE:
            metrics.consistent_discordance_count += 1

    # Correctness analysis (based on cell F1)
    correct_results = [r for r in valid_results if r.cell_f1 >= 0.5]
    incorrect_results = [r for r in valid_results if r.cell_f1 < 0.5]

    metrics.correct_samples_count = len(correct_results)
    metrics.incorrect_samples_count = len(incorrect_results)

    if correct_results:
        metrics.mean_confidence_when_correct = np.mean([
            r.aggregate_internal_confidence for r in correct_results
        ])
        metrics.mean_verbalized_when_correct = np.mean([
            float(r.verbalized_result.certainty_score)
            if r.verbalized_result is not None
            else float(r.aggregate_verbalized_certainty)
            for r in correct_results
        ])

    if incorrect_results:
        metrics.mean_confidence_when_incorrect = np.mean([
            r.aggregate_internal_confidence for r in incorrect_results
        ])
        metrics.mean_verbalized_when_incorrect = np.mean([
            float(r.verbalized_result.certainty_score)
            if r.verbalized_result is not None
            else float(r.aggregate_verbalized_certainty)
            for r in incorrect_results
        ])

    return metrics


def format_alignment_report(metrics: AlignmentMetrics) -> str:
    """
    Format alignment metrics as a readable report.

    Args:
        metrics: AlignmentMetrics object

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "CONFIDENCE-PROBABILITY ALIGNMENT REPORT",
        "=" * 70,
        "",
        f"Samples: {metrics.n_valid_samples} valid / {metrics.n_samples} total",
        "",
        "CORRELATION METRICS:",
        f"  Spearman's ρ:  {metrics.spearman_rho:+.4f}  (p={metrics.spearman_p_value:.4f})",
    ]

    if metrics.pearson_r is not None:
        lines.append(f"  Pearson's r:   {metrics.pearson_r:+.4f}  (p={metrics.pearson_p_value:.4f})")

    if metrics.kendall_tau is not None:
        lines.append(f"  Kendall's τ:   {metrics.kendall_tau:+.4f}  (p={metrics.kendall_p_value:.4f})")

    # Interpret correlation strength
    rho_abs = abs(metrics.spearman_rho)
    if rho_abs >= 0.7:
        strength = "STRONG"
    elif rho_abs >= 0.4:
        strength = "MODERATE"
    elif rho_abs >= 0.2:
        strength = "WEAK"
    else:
        strength = "NEGLIGIBLE"

    lines.extend([
        "",
        f"  Correlation Strength: {strength}",
        f"  Statistical Significance: {'YES' if metrics.spearman_p_value < 0.01 else 'NO'} (α=0.01)",
        "",
        "INTERNAL CONFIDENCE DISTRIBUTION:",
        f"  Mean:   {metrics.internal_mean:.4f}",
        f"  Std:    {metrics.internal_std:.4f}",
        f"  Median: {metrics.internal_median:.4f}",
        f"  Range:  [{metrics.internal_min:.4f}, {metrics.internal_max:.4f}]",
        "",
        "VERBALIZED CERTAINTY DISTRIBUTION:",
        f"  Mean:   {metrics.verbalized_mean:.4f}",
        f"  Std:    {metrics.verbalized_std:.4f}",
        f"  Median: {metrics.verbalized_median:.4f}",
        f"  Range:  [{metrics.verbalized_min:.4f}, {metrics.verbalized_max:.4f}]",
        "",
        "ALIGNMENT TYPE DISTRIBUTION:",
        f"  Consistent Alignment:     {metrics.consistent_alignment_count:4d} ({100*metrics.consistent_alignment_count/max(1,metrics.n_valid_samples):.1f}%)",
        f"  Internal Overconfidence:  {metrics.internal_overconfidence_count:4d} ({100*metrics.internal_overconfidence_count/max(1,metrics.n_valid_samples):.1f}%)",
        f"  External Overconfidence:  {metrics.external_overconfidence_count:4d} ({100*metrics.external_overconfidence_count/max(1,metrics.n_valid_samples):.1f}%)",
        f"  Consistent Discordance:   {metrics.consistent_discordance_count:4d} ({100*metrics.consistent_discordance_count/max(1,metrics.n_valid_samples):.1f}%)",
    ])

    if metrics.correct_samples_count > 0 or metrics.incorrect_samples_count > 0:
        lines.extend([
            "",
            "CONFIDENCE-CORRECTNESS RELATIONSHIP:",
            f"  Correct samples: {metrics.correct_samples_count}",
            f"    Mean Internal Confidence:  {metrics.mean_confidence_when_correct:.4f}",
            f"    Mean Verbalized Certainty: {metrics.mean_verbalized_when_correct:.4f}",
            f"  Incorrect samples: {metrics.incorrect_samples_count}",
            f"    Mean Internal Confidence:  {metrics.mean_confidence_when_incorrect:.4f}",
            f"    Mean Verbalized Certainty: {metrics.mean_verbalized_when_incorrect:.4f}",
        ])

    lines.append("=" * 70)

    return "\n".join(lines)


def alignment_metrics_to_dict(metrics: AlignmentMetrics) -> Dict[str, Any]:
    """
    Convert AlignmentMetrics to dictionary for JSON serialization.

    Args:
        metrics: AlignmentMetrics object

    Returns:
        Dictionary representation
    """
    return {
        "n_samples": metrics.n_samples,
        "n_valid_samples": metrics.n_valid_samples,
        "correlation": {
            "spearman_rho": round(metrics.spearman_rho, 4),
            "spearman_p_value": round(metrics.spearman_p_value, 6),
            "pearson_r": round(metrics.pearson_r, 4) if metrics.pearson_r else None,
            "pearson_p_value": round(metrics.pearson_p_value, 6) if metrics.pearson_p_value else None,
            "kendall_tau": round(metrics.kendall_tau, 4) if metrics.kendall_tau else None,
            "kendall_p_value": round(metrics.kendall_p_value, 6) if metrics.kendall_p_value else None,
        },
        "internal_confidence": {
            "mean": round(metrics.internal_mean, 4),
            "std": round(metrics.internal_std, 4),
            "median": round(metrics.internal_median, 4),
            "min": round(metrics.internal_min, 4),
            "max": round(metrics.internal_max, 4),
        },
        "verbalized_certainty": {
            "mean": round(metrics.verbalized_mean, 4),
            "std": round(metrics.verbalized_std, 4),
            "median": round(metrics.verbalized_median, 4),
            "min": round(metrics.verbalized_min, 4),
            "max": round(metrics.verbalized_max, 4),
        },
        "alignment_types": {
            "consistent_alignment": metrics.consistent_alignment_count,
            "internal_overconfidence": metrics.internal_overconfidence_count,
            "external_overconfidence": metrics.external_overconfidence_count,
            "consistent_discordance": metrics.consistent_discordance_count,
        },
        "correctness_relationship": {
            "correct_samples": metrics.correct_samples_count,
            "incorrect_samples": metrics.incorrect_samples_count,
            "mean_confidence_when_correct": round(metrics.mean_confidence_when_correct, 4),
            "mean_confidence_when_incorrect": round(metrics.mean_confidence_when_incorrect, 4),
            "mean_verbalized_when_correct": round(metrics.mean_verbalized_when_correct, 4),
            "mean_verbalized_when_incorrect": round(metrics.mean_verbalized_when_incorrect, 4),
        }
    }
