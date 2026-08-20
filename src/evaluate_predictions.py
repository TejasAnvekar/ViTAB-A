#!/usr/bin/env python3
"""
Evaluate VisualCite prediction JSONL files produced by vllm_inference.py.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from metrics import CellMetrics, aggregate_metrics, evaluate_single_prediction, format_metrics_table, metrics_to_dict


def _load_metrics(path: Path) -> tuple[List[CellMetrics], List[Dict[str, Any]]]:
    metrics: List[CellMetrics] = []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            metrics.append(
                evaluate_single_prediction(
                    predicted_cells=row.get("predicted_cells", []),
                    ground_truth_cells=row.get("ground_truth_cells", []),
                )
            )
    return metrics, rows


def evaluate_file(path: Path, output_dir: Path) -> Path:
    metrics, rows = _load_metrics(path)
    agg = aggregate_metrics(metrics)
    errored = sum(1 for row in rows if row.get("error"))
    structured = sum(1 for row in rows if row.get("parsed_structured"))
    avg_latency = (
        sum(float(row.get("inference_time_ms", 0.0)) for row in rows) / len(rows)
        if rows
        else 0.0
    )
    avg_mean_logprob = [
        float(row["mean_token_logprob"])
        for row in rows
        if row.get("mean_token_logprob") is not None
    ]

    summary = {
        "input_path": str(path),
        "total_records": len(rows),
        "errored_records": errored,
        "structured_parse_rate": structured / len(rows) if rows else 0.0,
        "avg_inference_time_ms": avg_latency,
        "avg_mean_token_logprob": (
            sum(avg_mean_logprob) / len(avg_mean_logprob) if avg_mean_logprob else None
        ),
        "metrics": metrics_to_dict(agg),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(format_metrics_table(agg))
    print(f"\nErrored records: {errored}/{len(rows)}")
    print(f"Structured parse rate: {summary['structured_parse_rate']:.4f}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Saved summary: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vLLM VisualCite prediction logs")
    parser.add_argument("prediction_jsonl", help="JSONL file produced by vllm_inference.py")
    parser.add_argument("--output-dir", default="vllm_results/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_file(Path(args.prediction_jsonl), Path(args.output_dir))


if __name__ == "__main__":
    main()
