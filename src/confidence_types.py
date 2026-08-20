"""
Data types for Confidence-Probability Alignment evaluation.

Based on the paper "Confidence Under the Hood: An Investigation into the
Confidence-Probability Alignment in Large Language Models" (ACL 2024).

This module defines the core data structures for:
- ConfidenceResult: Stores internal confidence (from token probabilities) and
  verbalized certainty (from follow-up query)
- AlignmentMetrics: Stores Spearman's rho and other alignment statistics
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class CertaintyLevel(Enum):
    """
    Likert scale for verbalized certainty (from paper Section 2.3).
    Maps to numerical scores for alignment computation.
    """
    VERY_CERTAIN = ("a", "very certain", 1.0)
    FAIRLY_CERTAIN = ("b", "fairly certain", 0.8)
    MODERATELY_CERTAIN = ("c", "moderately certain", 0.6)
    SOMEWHAT_CERTAIN = ("d", "somewhat certain", 0.4)
    NOT_CERTAIN = ("e", "not certain", 0.2)
    VERY_UNCERTAIN = ("f", "very uncertain", 0.0)

    def __init__(self, letter: str, text: str, score: float):
        self.letter = letter
        self.text = text
        self.score = score

    @classmethod
    def from_response(cls, response: str) -> Optional['CertaintyLevel']:
        """
        Parse model response to extract certainty level.
        Handles various response formats (letter only, text, or combination).
        """
        response_lower = response.lower().strip()

        # Try to match by letter (e.g., "a", "a.", "a)")
        for level in cls:
            if response_lower.startswith(level.letter + ".") or \
               response_lower.startswith(level.letter + ")") or \
               response_lower == level.letter:
                return level

        # Try to match by text
        for level in cls:
            if level.text in response_lower:
                return level

        # Try partial matches
        if "very certain" in response_lower and "uncertain" not in response_lower:
            return cls.VERY_CERTAIN
        if "fairly certain" in response_lower:
            return cls.FAIRLY_CERTAIN
        if "moderately certain" in response_lower:
            return cls.MODERATELY_CERTAIN
        if "somewhat certain" in response_lower:
            return cls.SOMEWHAT_CERTAIN
        if "not certain" in response_lower:
            return cls.NOT_CERTAIN
        if "very uncertain" in response_lower or "uncertain" in response_lower:
            return cls.VERY_UNCERTAIN

        return None


class AlignmentType(Enum):
    """
    Types of alignment/misalignment between internal confidence and verbalized certainty.
    From paper Table 2.
    """
    CONSISTENT_ALIGNMENT = "consistent_alignment"  # Both high or both low
    INTERNAL_OVERCONFIDENCE = "internal_overconfidence"  # High internal, low verbalized
    EXTERNAL_OVERCONFIDENCE = "external_overconfidence"  # Low internal, high verbalized
    CONSISTENT_DISCORDANCE = "consistent_discordance"  # Both low (model uncertain)


@dataclass
class OptionProbability:
    """Token probability for a single answer option."""
    option_label: str  # e.g., "A", "B", "C"
    option_text: str   # e.g., "Cell A1", "Cell B2"
    raw_probability: float
    adjusted_probability: float  # After normalization across all options
    token_id: Optional[int] = None
    log_probability: Optional[float] = None


@dataclass
class InternalConfidenceResult:
    """
    Internal confidence derived from token probabilities.

    Following Algorithm 1 from the paper:
    1. Convert log probabilities to probabilities
    2. For each option, find the maximum probability among corresponding tokens
    3. Normalize the selected answer's probability against sum of all option probabilities
    """
    # The model's selected answer
    selected_option: str  # e.g., "A"
    selected_option_text: str  # e.g., "Cell A1"

    # Raw internal confidence (probability of selected token)
    raw_confidence: float

    # Adjusted internal confidence (P_IC from Algorithm 1)
    adjusted_confidence: float

    # Per-option probabilities
    option_probabilities: List[OptionProbability] = field(default_factory=list)

    # Metadata
    total_options: int = 0
    answer_token_ambiguity: bool = False  # e.g., 'B' vs 'b' issue


@dataclass
class VerbalizedCertaintyResult:
    """
    Verbalized certainty from follow-up confidence query.

    Uses the Confidence Querying Prompt (CQP) design from paper Section 2.3:
    - Third-Person Perspective (TPP)
    - Option Contextualization (OC)
    - Likert Scale Utilization (LSU)
    """
    # The certainty level selected by the model
    certainty_level: Optional[CertaintyLevel]

    # Numerical score (0.0 to 1.0)
    certainty_score: float

    # Raw model response to the confidence query
    raw_response: str

    # Whether the response was successfully parsed
    parse_success: bool

    # The confidence querying prompt used
    query_prompt: Optional[str] = None


@dataclass
class ConfidenceResult:
    """
    Combined confidence result for a single sample.

    Stores both:
    - Internal confidence (from token probabilities during answer generation)
    - Verbalized certainty (from follow-up query asking model about its confidence)
    """
    # Sample identification
    sample_id: str

    # The question asked
    question: str

    # The answer options provided
    answer_options: List[str]

    # The model's selected answer
    selected_answer: str
    selected_answer_index: int

    # Ground truth (if available)
    ground_truth_answer: Optional[str] = None
    is_correct: Optional[bool] = None

    # Internal confidence from token probabilities
    internal_confidence: Optional[InternalConfidenceResult] = None

    # Verbalized certainty from follow-up query
    verbalized_certainty: Optional[VerbalizedCertaintyResult] = None

    # Alignment classification
    alignment_type: Optional[AlignmentType] = None

    # Timing information
    answer_inference_time_ms: float = 0.0
    confidence_query_time_ms: float = 0.0

    # Additional metadata
    model_name: str = ""
    representation: str = ""
    strategy: str = ""

    def compute_alignment_type(self, threshold: float = 0.5) -> AlignmentType:
        """
        Determine the alignment type based on internal and verbalized confidence.

        Args:
            threshold: Threshold to classify high vs low confidence (default 0.5)

        Returns:
            AlignmentType classification
        """
        if self.internal_confidence is None or self.verbalized_certainty is None:
            return None

        internal = self.internal_confidence.adjusted_confidence
        verbalized = self.verbalized_certainty.certainty_score

        internal_high = internal >= threshold
        verbalized_high = verbalized >= threshold

        if internal_high and verbalized_high:
            return AlignmentType.CONSISTENT_ALIGNMENT
        elif internal_high and not verbalized_high:
            return AlignmentType.INTERNAL_OVERCONFIDENCE
        elif not internal_high and verbalized_high:
            return AlignmentType.EXTERNAL_OVERCONFIDENCE
        else:
            return AlignmentType.CONSISTENT_DISCORDANCE


@dataclass
class AlignmentMetrics:
    """
    Alignment statistics computed across multiple samples.

    Primary metric is Spearman's rank correlation coefficient (rho),
    as described in paper Section 2.4.
    """
    # Primary alignment metric
    spearman_rho: float = 0.0
    spearman_p_value: float = 0.0

    # Additional correlation metrics
    pearson_r: Optional[float] = None
    pearson_p_value: Optional[float] = None
    kendall_tau: Optional[float] = None
    kendall_p_value: Optional[float] = None

    # Sample statistics
    n_samples: int = 0
    n_valid_samples: int = 0  # Samples with both confidence measures

    # Distribution statistics for internal confidence
    internal_mean: float = 0.0
    internal_std: float = 0.0
    internal_median: float = 0.0
    internal_min: float = 0.0
    internal_max: float = 0.0

    # Distribution statistics for verbalized certainty
    verbalized_mean: float = 0.0
    verbalized_std: float = 0.0
    verbalized_median: float = 0.0
    verbalized_min: float = 0.0
    verbalized_max: float = 0.0

    # Alignment type distribution
    consistent_alignment_count: int = 0
    internal_overconfidence_count: int = 0
    external_overconfidence_count: int = 0
    consistent_discordance_count: int = 0

    # Correctness-confidence relationship
    correct_samples_count: int = 0
    incorrect_samples_count: int = 0
    mean_confidence_when_correct: float = 0.0
    mean_confidence_when_incorrect: float = 0.0
    mean_verbalized_when_correct: float = 0.0
    mean_verbalized_when_incorrect: float = 0.0

    # Per-sample results for detailed analysis
    sample_results: List[ConfidenceResult] = field(default_factory=list)


@dataclass
class CellConfidenceResult:
    """
    Confidence result specifically for cell attribution task.

    Extends the general confidence framework to handle:
    - Multiple cells as answer options
    - Cell-level confidence scores
    """
    # Sample identification
    sample_id: str
    question: str
    answer: str

    # All cells in the table as options
    all_cells: List[str]  # e.g., ["A1", "A2", "B1", "B2", ...]

    # Predicted cells and their confidences
    predicted_cells: List[str]
    cell_confidences: Dict[str, float]  # cell -> confidence score

    # Ground truth cells
    ground_truth_cells: List[str]

    # Aggregate confidence for the prediction
    aggregate_internal_confidence: Optional[float] = None
    aggregate_verbalized_certainty: Optional[float] = None

    # Verbalized certainty result
    verbalized_result: Optional[VerbalizedCertaintyResult] = None

    # Correctness metrics
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    exact_match: bool = False

    # Timing
    inference_time_ms: float = 0.0
    confidence_query_time_ms: float = 0.0

    # Metadata
    model_name: str = ""
    representation: str = ""


def classify_alignment(
    internal_confidence: float,
    verbalized_certainty: float,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4
) -> AlignmentType:
    """
    Classify the alignment type between internal and verbalized confidence.

    Args:
        internal_confidence: Internal confidence score (0-1)
        verbalized_certainty: Verbalized certainty score (0-1)
        high_threshold: Threshold for "high" confidence
        low_threshold: Threshold for "low" confidence

    Returns:
        AlignmentType classification
    """
    internal_high = internal_confidence >= high_threshold
    verbalized_high = verbalized_certainty >= high_threshold
    internal_low = internal_confidence <= low_threshold
    verbalized_low = verbalized_certainty <= low_threshold

    if internal_high and verbalized_high:
        return AlignmentType.CONSISTENT_ALIGNMENT
    elif internal_high and verbalized_low:
        return AlignmentType.INTERNAL_OVERCONFIDENCE
    elif internal_low and verbalized_high:
        return AlignmentType.EXTERNAL_OVERCONFIDENCE
    elif internal_low and verbalized_low:
        return AlignmentType.CONSISTENT_DISCORDANCE
    else:
        # Intermediate case - classify based on relative difference
        diff = internal_confidence - verbalized_certainty
        if abs(diff) < 0.2:
            return AlignmentType.CONSISTENT_ALIGNMENT
        elif diff > 0:
            return AlignmentType.INTERNAL_OVERCONFIDENCE
        else:
            return AlignmentType.EXTERNAL_OVERCONFIDENCE
