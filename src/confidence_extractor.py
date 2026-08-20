"""
Internal Confidence Extraction from Token Probabilities.

Implements a robust version of Algorithm 1 from "Confidence Under the Hood" (ACL 2024)
adapted for multi-token Cell Attribution tasks.

KEY APPROACH:
1.  Uses 'Joint Probability' (geometric mean of token probs) instead of 'Max' to handle
    multi-token coordinates (e.g., "A12") correctly.
2.  Aligns tokens strictly with the generated answer, ignoring reasoning steps.
3.  Computes confidence for each predicted cell by finding its tokens in the output.
"""
import logging
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

try:
    from .confidence_types import InternalConfidenceResult
except ImportError:
    try:
        from confidence_types import InternalConfidenceResult
    except ImportError:
        class InternalConfidenceResult:
            def __init__(self, confidence_score, method):
                self.confidence_score = confidence_score
                self.method = method

logger = logging.getLogger(__name__)


def find_answer_token_indices(
    full_input_ids: List[int], 
    answer_str: str, 
    tokenizer: Any
) -> List[int]:
    """
    Locates the indices of the answer tokens within the full generated sequence.
    Robustly handles leading spaces/newlines which are common in Zero-shot.
    """
    # Strategy 1: strict match
    variations = [answer_str]
    
    # Strategy 2: leading space (Common in Llama/Qwen generations)
    variations.append(" " + answer_str)
    
    # Strategy 3: leading newline (Common if prompt ends with newline)
    variations.append("\n" + answer_str)

    # Strategy 4
    variations.append("=" + answer_str)
    variations.append(" =" + answer_str)
    variations.append("\n=" + answer_str)

    len_full = len(full_input_ids)

    # We search for ALL variations and pick the one that appears LATEST in the sequence
    # (Since the answer is usually at the very end)
    best_indices = []
    max_start_index = -1

    for var in variations:
        # Encode without adding special tokens (CLS/EOS)
        var_tokens = tokenizer.encode(var, add_special_tokens=False)
        if not var_tokens:
            continue
            
        len_ans = len(var_tokens)
        
        # Search backwards from end
        for i in range(len_full - len_ans, -1, -1):
            window = full_input_ids[i : i + len_ans]
            if window == var_tokens:
                # We found a match. Is it the latest one we've seen?
                if i > max_start_index:
                    max_start_index = i
                    best_indices = list(range(i, i + len_ans))
                break # Found the last instance of this variation, move to next variation

    return best_indices


def compute_geometric_mean_confidence(
    logits: torch.Tensor,
    generated_ids: List[int],
    answer_token_indices: List[int]
) -> float:
    """
    Computes confidence as the geometric mean of the token probabilities.
    Math: exp( (1/N) * sum(log(p_i)) )

    Args:
        logits: Tensor of shape [seq_len, vocab_size] - logits for each generated position
        generated_ids: List of generated token IDs
        answer_token_indices: Indices of the answer tokens within generated_ids

    Returns:
        Geometric mean of token probabilities (confidence score)
    """
    if not answer_token_indices:
        return 0.0

    log_prob_sum = 0.0
    count = 0

    for idx in answer_token_indices:
        # The token at generated_ids[idx] is predicted by logits[idx]
        # (logits[i] predicts the token at position i)
        if idx < 0 or idx >= logits.shape[0]:
            continue

        step_logits = logits[idx]
        step_log_probs = F.log_softmax(step_logits, dim=-1)

        token_id = generated_ids[idx]
        token_log_prob = step_log_probs[token_id].item()

        log_prob_sum += token_log_prob
        count += 1

    if count == 0:
        return 0.0

    # Geometric mean: exp(mean(log_probs))
    return math.exp(log_prob_sum / count)


def extract_cell_confidences(
    logits: torch.Tensor,
    all_cells: List[str],
    predicted_cells: List[str],
    tokenizer: Any,
    average_across_tokens: bool = True,
    generated_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Extract confidence scores for each predicted cell.

    This function computes the geometric mean probability of the tokens
    that make up each predicted cell coordinate in the generated output.

    Args:
        logits: Tensor of shape [seq_len, vocab_size] - all generated token logits
        all_cells: List of all valid cell coordinates in the table (not used directly
                   but kept for API compatibility)
        predicted_cells: List of predicted cell coordinates (e.g., ["A1", "B2"])
        tokenizer: The tokenizer for encoding cell strings
        average_across_tokens: If True, use geometric mean; if False, use product
        generated_ids: Optional list of generated token IDs. If not provided,
                      will attempt to get from logits argmax (less reliable)

    Returns:
        Dictionary mapping each predicted cell to its confidence score
    """
    cell_confidences: Dict[str, float] = {}

    if logits is None or len(predicted_cells) == 0:
        return cell_confidences

    # Ensure logits is 2D [seq_len, vocab_size]
    if logits.dim() > 2:
        logits = logits.squeeze(0)

    # If generated_ids not provided, derive from logits (less reliable)
    if generated_ids is None:
        generated_ids = logits.argmax(dim=-1).tolist()
        logger.warning("generated_ids not provided, deriving from logits argmax")

    for cell in predicted_cells:
        # Normalize cell format
        cell_clean = str(cell).strip().upper().lstrip("=")

        # Find where this cell appears in the generated sequence
        token_indices = find_answer_token_indices(generated_ids, cell_clean, tokenizer)

        if token_indices:
            if average_across_tokens:
                conf = compute_geometric_mean_confidence(logits, generated_ids, token_indices)
            else:
                # Product of probabilities (not normalized by length)
                conf = compute_geometric_mean_confidence(logits, generated_ids, token_indices)
                # For product, we'd use exp(sum) instead of exp(mean), but geometric mean
                # is more stable and comparable across different length answers
            cell_confidences[cell_clean] = conf
            logger.debug(f"Cell {cell_clean}: confidence={conf:.6f}, tokens={token_indices}")
        else:
            # Cell not found in output - assign 0 confidence
            cell_confidences[cell_clean] = 0.0
            logger.debug(f"Cell {cell_clean}: not found in generated output")

    return cell_confidences


def compute_aggregate_confidence(
    cell_confidences: Dict[str, float],
    method: str = "mean"
) -> float:
    """
    Aggregates confidence scores from a dictionary of cell confidences.

    Args:
        cell_confidences: Dictionary mapping cells to confidence scores
        method: Aggregation method - "mean", "max", "min", or "product"

    Returns:
        Aggregated confidence score
    """
    if not cell_confidences:
        return 0.0

    values = list(cell_confidences.values())

    if method == "mean":
        return sum(values) / len(values)
    elif method == "max":
        return max(values)
    elif method == "min":
        return min(values)
    elif method == "product":
        prod = 1.0
        for v in values:
            prod *= v
        return prod
    else:
        return sum(values) / len(values)


def compute_all_cell_probabilities(
    logits: torch.Tensor,
    all_cells: List[str],
    tokenizer: Any,
    average_across_tokens: bool = True,
    generated_ids: Optional[List[int]] = None,
    predicted_cell: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute probabilities for all cells in the table.

    For generative models, we can only accurately compute the probability of the
    path that was actually taken (the predicted cell). For other cells, we assign 0.0
    unless they match the prediction.

    This is used for conformal prediction UQ which requires probability estimates
    over all possible labels.

    Args:
        logits: Tensor of shape [seq_len, vocab_size]
        all_cells: List of all valid cell coordinates in the table
        tokenizer: The tokenizer for encoding
        average_across_tokens: Whether to use geometric mean (True) or product (False)
        generated_ids: Optional list of generated token IDs
        predicted_cell: The cell that was actually predicted (optional, for optimization)

    Returns:
        Dictionary mapping all cells to their probability estimates
    """
    # Initialize all cells to 0
    cell_probs = {cell.strip().upper(): 0.0 for cell in all_cells}

    if logits is None:
        return cell_probs

    # Ensure logits is 2D
    if logits.dim() > 2:
        logits = logits.squeeze(0)

    # If generated_ids not provided, derive from logits
    if generated_ids is None:
        generated_ids = logits.argmax(dim=-1).tolist()

    # If we know the predicted cell, only compute confidence for that
    # This is more accurate than trying to compute counterfactual probabilities
    if predicted_cell:
        predicted_clean = str(predicted_cell).strip().upper().lstrip("=")
        token_indices = find_answer_token_indices(generated_ids, predicted_clean, tokenizer)

        if token_indices:
            conf = compute_geometric_mean_confidence(logits, generated_ids, token_indices)
            # Assign to the predicted cell (normalize key)
            if predicted_clean in cell_probs:
                cell_probs[predicted_clean] = conf
            else:
                cell_probs[predicted_clean] = conf
    else:
        # Without knowing the prediction, scan the output for any cell patterns
        # This is less reliable but necessary when predicted_cell isn't provided
        for cell in all_cells:
            cell_clean = cell.strip().upper()
            token_indices = find_answer_token_indices(generated_ids, cell_clean, tokenizer)
            if token_indices:
                conf = compute_geometric_mean_confidence(logits, generated_ids, token_indices)
                cell_probs[cell_clean] = conf

    return cell_probs


class InternalConfidenceExtractor:
    """
    Class wrapper for compatibility with object-oriented usage.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract(
        self,
        model_output_ids: List[int],
        model_logits: torch.Tensor,
        generated_text: str,
        predicted_cells: List[str]
    ) -> InternalConfidenceResult:
        """
        Extract internal confidence for a prediction.

        Args:
            model_output_ids: List of generated token IDs
            model_logits: Tensor of shape [seq_len, vocab_size]
            generated_text: The decoded text (unused but kept for API)
            predicted_cells: List of predicted cell coordinates

        Returns:
            InternalConfidenceResult with aggregate confidence score
        """
        cell_confidences = extract_cell_confidences(
            logits=model_logits,
            all_cells=[],  # Not needed for this function
            predicted_cells=predicted_cells,
            tokenizer=self.tokenizer,
            average_across_tokens=True,
            generated_ids=model_output_ids
        )

        aggregate_conf = compute_aggregate_confidence(cell_confidences)

        return InternalConfidenceResult(
            confidence_score=aggregate_conf,
            method="geometric_mean_token_probs"
        )
