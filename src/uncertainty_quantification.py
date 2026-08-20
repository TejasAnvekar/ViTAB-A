"""Uncertainty quantification utilities for VisualCite benchmarking.

This module implements split-conformal prediction over discrete answer options
using first-token probabilities (already computed under the hood).

We provide two common nonconformity scores:
- LAC (Least Ambiguous Class): score = 1 - p(y)
- APS (Adaptive Prediction Sets): score = cumulative probability mass up to y

Notes for VisualCite:
- Labels are table cell coordinates (e.g., "A1", "E7").
- Ground truth may contain multiple cells; coverage is counted as success if
  any ground-truth cell is contained in the predicted set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ConformalResult:
    method: str  # "LAC" or "APS"
    alpha: float
    calibration_ratio: float
    n_cal: int
    n_test: int
    qhat: float
    coverage: float
    avg_set_size: float
    top1_hit_rate: float


def _stable_train_test_split_indices(
    n: int,
    train_size: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_size < 1.0):
        raise ValueError(f"train_size must be in (0,1), got {train_size}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(train_size * n))
    if n_train <= 0 or n_train >= n:
        raise ValueError(f"train_size produced degenerate split: n={n}, n_train={n_train}")
    return perm[:n_train], perm[n_train:]


def _normalize_cell(cell: str) -> str:
    return str(cell).strip().upper().lstrip("=")


def _get_truth_cells(truth_cells: Sequence[str]) -> List[str]:
    return [_normalize_cell(c) for c in truth_cells if str(c).strip()]


def _sorted_probs(cell_probs: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, probs) sorted descending by prob."""
    if not cell_probs:
        return np.array([], dtype=object), np.array([], dtype=float)
    labels = np.array([_normalize_cell(k) for k in cell_probs.keys()], dtype=object)
    probs = np.array(list(cell_probs.values()), dtype=float)
    order = np.argsort(probs)[::-1]
    return labels[order], probs[order]


def _top1_label(cell_probs: Dict[str, float]) -> Optional[str]:
    if not cell_probs:
        return None
    return _normalize_cell(max(cell_probs.items(), key=lambda kv: kv[1])[0])


def _lac_score(cell_probs: Dict[str, float], truth_cells: Sequence[str]) -> Optional[float]:
    truths = _get_truth_cells(truth_cells)
    if not truths or not cell_probs:
        return None
    # For multi-label truth, use best (max prob) truth to match "any-hit" coverage.
    p_truth = max(float(cell_probs.get(t, 0.0)) for t in truths)
    return 1.0 - p_truth


def _aps_score(cell_probs: Dict[str, float], truth_cells: Sequence[str]) -> Optional[float]:
    truths = _get_truth_cells(truth_cells)
    if not truths or not cell_probs:
        return None

    labels_sorted, probs_sorted = _sorted_probs(cell_probs)
    if probs_sorted.size == 0:
        return None
    cumsum = np.cumsum(probs_sorted)

    # For multi-label truth, use the *minimum* cumulative mass among truth labels.
    # This aligns with "coverage" being satisfied if any truth is in the set.
    best_score: Optional[float] = None
    label_to_index = {str(lbl): int(i) for i, lbl in enumerate(labels_sorted.tolist())}
    for t in truths:
        idx = label_to_index.get(t)
        if idx is None:
            continue
        score = float(cumsum[idx])
        if best_score is None or score < best_score:
            best_score = score
    return best_score


def _conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    # Standard split conformal: q_level = ceil((n+1)*(1-alpha))/n
    n = len(scores)
    if n == 0:
        raise ValueError("No calibration scores")
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(np.asarray(scores, dtype=float), q_level, method="higher"))


def _predict_set_lac(cell_probs: Dict[str, float], qhat: float) -> List[str]:
    if not cell_probs:
        return []
    threshold = 1.0 - qhat
    labels_sorted, probs_sorted = _sorted_probs(cell_probs)
    selected = [str(lbl) for lbl, p in zip(labels_sorted.tolist(), probs_sorted.tolist()) if float(p) >= threshold]
    if not selected:
        top1 = _top1_label(cell_probs)
        return [top1] if top1 is not None else []
    return selected


def _predict_set_aps(cell_probs: Dict[str, float], qhat: float) -> List[str]:
    if not cell_probs:
        return []
    labels_sorted, probs_sorted = _sorted_probs(cell_probs)
    if probs_sorted.size == 0:
        return []

    cumsum = np.cumsum(probs_sorted)
    selected: List[str] = []
    for lbl, cs in zip(labels_sorted.tolist(), cumsum.tolist()):
        if float(cs) <= qhat:
            selected.append(str(lbl))
        else:
            break

    if not selected:
        top1 = _top1_label(cell_probs)
        return [top1] if top1 is not None else []
    return selected


def run_split_conformal(
    instances: Sequence[Dict[str, object]],
    calibration_ratio: float = 0.5,
    alpha: float = 0.1,
    seed: int = 42,
) -> Dict[str, ConformalResult]:
    """Run LAC and APS split conformal prediction.

    Args:
        instances: Each item must include:
            - "sample_id": str
            - "ground_truth_cells": List[str]
            - "all_cell_probabilities": Dict[str, float]
        calibration_ratio: Fraction of instances used for calibration.
        alpha: Miscoverage level (target coverage is 1 - alpha).
        seed: RNG seed for split.

    Returns:
        Dict with keys "LAC" and "APS" containing summary results.
    """
    # Filter to instances with required fields
    usable = []
    for inst in instances:
        probs = inst.get("all_cell_probabilities")
        truths = inst.get("ground_truth_cells")
        if isinstance(probs, dict) and isinstance(truths, list) and probs and truths:
            usable.append(inst)

    n = len(usable)
    if n < 2:
        raise ValueError(f"Need at least 2 usable instances for split conformal, got {n}")

    cal_idx, test_idx = _stable_train_test_split_indices(n, calibration_ratio, seed)
    cal = [usable[int(i)] for i in cal_idx.tolist()]
    test = [usable[int(i)] for i in test_idx.tolist()]

    def compute(method: str) -> ConformalResult:
        # Calibration scores
        scores: List[float] = []
        for inst in cal:
            probs = inst["all_cell_probabilities"]  # type: ignore[index]
            truths = inst["ground_truth_cells"]  # type: ignore[index]
            if method == "LAC":
                s = _lac_score(probs, truths)  # type: ignore[arg-type]
            elif method == "APS":
                s = _aps_score(probs, truths)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown method: {method}")
            if s is not None:
                scores.append(float(s))

        qhat = _conformal_quantile(scores, alpha)

        # Test prediction sets
        cover = []
        sizes = []
        top1_hits = []
        for inst in test:
            probs = inst["all_cell_probabilities"]  # type: ignore[index]
            truths = _get_truth_cells(inst["ground_truth_cells"])  # type: ignore[index]
            if method == "LAC":
                pred_set = _predict_set_lac(probs, qhat)  # type: ignore[arg-type]
            else:
                pred_set = _predict_set_aps(probs, qhat)  # type: ignore[arg-type]

            pred_set_norm = set(_normalize_cell(x) for x in pred_set)
            sizes.append(len(pred_set_norm))

            hit = any(t in pred_set_norm for t in truths)
            cover.append(1.0 if hit else 0.0)

            top1 = _top1_label(probs)  # type: ignore[arg-type]
            top1_hit = (top1 is not None) and any(top1 == t for t in truths)
            top1_hits.append(1.0 if top1_hit else 0.0)

        return ConformalResult(
            method=method,
            alpha=float(alpha),
            calibration_ratio=float(calibration_ratio),
            n_cal=len(cal),
            n_test=len(test),
            qhat=float(qhat),
            coverage=float(np.mean(cover)) if cover else 0.0,
            avg_set_size=float(np.mean(sizes)) if sizes else 0.0,
            top1_hit_rate=float(np.mean(top1_hits)) if top1_hits else 0.0,
        )

    return {
        "LAC": compute("LAC"),
        "APS": compute("APS"),
    }
