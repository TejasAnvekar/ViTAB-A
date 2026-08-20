"""
Configuration for VisualCite Attribution Benchmark
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ModelSize(Enum):
    """Model identifiers for benchmark runs"""
    # Gemma family
    GEMMA_4B = "google/gemma-3-4b-it"
    GEMMA_12B = "google/gemma-3-12b-it"
    GEMMA_27B = "google/gemma-3-27b-it"
    # InternVL 3.5 family
    INTERNVL35_4B = "OpenGVLab/InternVL3_5-4B-hf"
    INTERNVL35_8B = "OpenGVLab/InternVL3_5-8B-hf"
    INTERNVL35_14B = "OpenGVLab/InternVL3_5-14B-hf"
    INTERNVL35_38B = "OpenGVLab/InternVL3_5-38B-hf"
    # Qwen3-VL family
    VL_2B = "Qwen/Qwen3-VL-2B-Instruct"
    VL_4B = "Qwen/Qwen3-VL-4B-Instruct"
    VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
    VL_32B = "Qwen/Qwen3-VL-32B-Instruct"
    # Molmo2 family
    MOLMO2_4B = "allenai/Molmo2-4B"
    MOLMO2_8B = "allenai/Molmo2-8B"


class DataRepresentation(Enum):
    """Table data representation types"""
    JSON = "json"
    MARKDOWN = "markdown"
    IMAGE_ARIAL = "image_arial"
    IMAGE_TIMES = "image_times_new_roman"
    IMAGE_RED = "image_red"
    IMAGE_BLUE = "image_blue"
    IMAGE_GREEN = "image_green"


class PromptStrategy(Enum):
    """Prompting strategies for attribution"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration"""
    # Dataset settings
    jsonl_path: str = "../visualcite.jsonl"  # Path to local JSONL file
    dataset_split: str = "dev"  # Use dev split for benchmarking
    max_samples: Optional[int] = 200  # Limit to 200 samples
    single_cell_only: bool = True  # Only load samples with exactly one ground truth cell

    # Model settings
    models: List[str] = field(default_factory=lambda: [
        ModelSize.VL_2B.value,
        ModelSize.VL_4B.value,
        ModelSize.VL_8B.value,
        ModelSize.VL_32B.value,
    ])

    # Data representations to test
    representations: List[DataRepresentation] = field(default_factory=lambda: [
        DataRepresentation.JSON,
        DataRepresentation.MARKDOWN,
        DataRepresentation.IMAGE_ARIAL,
        DataRepresentation.IMAGE_TIMES,
        DataRepresentation.IMAGE_RED,
        DataRepresentation.IMAGE_BLUE,
        DataRepresentation.IMAGE_GREEN,
    ])

    # Prompt strategies
    strategies: List[PromptStrategy] = field(default_factory=lambda: [
        PromptStrategy.ZERO_SHOT,
        PromptStrategy.FEW_SHOT,
        PromptStrategy.CHAIN_OF_THOUGHT,
    ])

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False

    # Confidence extraction settings (optional)
    extract_internal_confidence: bool = True
    extract_verbalized_certainty: bool = True
    cqp_template: str = "cell_attribution"
    cqp_max_tokens: int = 32
    cqp_temperature: float = 0.0

    # Uncertainty quantification (split conformal prediction)
    enable_conformal_uq: bool = True
    conformal_calibration_ratio: float = 0.5
    conformal_alpha: float = 0.1
    conformal_seed: int = 42

    # Output settings
    output_dir: str = "benchmark_results"
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"

    # Hardware settings
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    use_torch_compile: bool = False

    # Checkpoint/resume settings
    resume_from_checkpoint: bool = True
    checkpoint_every_n_samples: int = 50


# Prompt templates
PROMPT_TEMPLATES = {
    PromptStrategy.ZERO_SHOT: """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

TASK: Identify the cell coordinate(s) that contain or directly support this answer. Use Excel-style coordinates where columns are letters (A, B, C, ...) and rows are numbers (1, 2, 3, ...).

RESPONSE FORMAT: Return ONLY the cell coordinates in Excel formula format. Examples:
- Single cell: "=E7" or "=B3"
- Multiple cells: "=A2" or list them separately: "=A2, =B2, =C2"
- If the answer involves a formula (sum, average, etc.), you may use: "SUM(C3:C10)" or "=A1+B2"

IMPORTANT: Do NOT repeat the question, table, or instructions. Output ONLY the cell coordinates.

ATTRIBUTED CELLS:""",

    PromptStrategy.FEW_SHOT: """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

Here is an example:

EXAMPLE:
TABLE:
{example1_table}
QUESTION: {example1_question}
ANSWER: {example1_answer}
ATTRIBUTED CELLS: {example1_cells}

Now analyze this table:

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

IMPORTANT: Do NOT repeat the example, question, table, or instructions. Output ONLY the cell coordinates in formula format.

ATTRIBUTED CELLS:""",

    PromptStrategy.CHAIN_OF_THOUGHT: """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

Let's think step by step:

1. First, understand what the question is asking for.
2. Then, locate where the answer "{answer}" appears or can be derived from in the table.
3. Identify the specific cell coordinate(s) using Excel-style notation (columns as letters A, B, C... and rows as numbers 1, 2, 3...).
4. If the answer is computed from multiple cells (e.g., a sum), express it as a formula like "SUM(C3:C10)" or "=A1+B2".
5. For simple cell references, use the format "=E7" or "=B3".

IMPORTANT: Do NOT repeat the question or table in your reasoning.

REASONING:
"""
}


# Additional CoT suffix to extract final answer
COT_EXTRACTION_SUFFIX = """

Based on the above reasoning, provide ONLY the final cell coordinates in Excel formula format (e.g., "=E7", "SUM(C3:C10)", or "=A1+B2"). Do NOT repeat your reasoning or the question.
ATTRIBUTED CELLS:"""
