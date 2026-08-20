"""
VisualCite Attribution Benchmark

Unified benchmark for vision-language models on cell attribution task.
Supports: Gemma, InternVL 3.5, Qwen3-VL, and Molmo2.
"""

__version__ = "2.0.0"

from .config import BenchmarkConfig, DataRepresentation, PromptStrategy, ModelSize
from .data_loader import VisualCiteDataset, VisualCiteSample
from .metrics import evaluate_single_prediction, aggregate_metrics, CellMetrics, AggregatedMetrics
from .model_handler import InternVLModel, Qwen3VLModel, ModelManager
from .prompt_builder import PromptBuilder
from .checkpoint_manager import CheckpointManager
from .result_logger import ResultLogger

__all__ = [
    "BenchmarkConfig",
    "DataRepresentation",
    "PromptStrategy",
    "ModelSize",
    "VisualCiteDataset",
    "VisualCiteSample",
    "evaluate_single_prediction",
    "aggregate_metrics",
    "CellMetrics",
    "AggregatedMetrics",
    "InternVLModel",
    "Qwen3VLModel",
    "ModelManager",
    "PromptBuilder",
    "CheckpointManager",
    "ResultLogger"
]
