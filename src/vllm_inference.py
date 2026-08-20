#!/usr/bin/env python3
"""
Async OpenAI-compatible vLLM inference for VisualCite text-table attribution.

Start a server separately, for example:

    vllm serve Qwen/Qwen3-VL-2B-Instruct --served-model-name vitab-model

Then run:

    python vllm_inference.py --model vitab-model --representation markdown
"""

import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from config import DataRepresentation, PromptStrategy
from data_loader import VisualCiteDataset, parse_model_output
from metrics import evaluate_single_prediction
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class AttributionPrediction(BaseModel):
    """Structured output requested from the model."""

    cells: List[str] = Field(
        default_factory=list,
        description="Excel-style cells that support the answer, for example ['E7'].",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Short optional rationale. Keep empty unless needed.",
    )


@dataclass
class VLLMRecord:
    sample_id: str
    model_name: str
    representation: str
    strategy: str
    question: str
    answer: str
    ground_truth_cells: List[str]
    predicted_cells: List[str]
    parsed_structured: bool
    raw_output: str
    prompt: str
    inference_time_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    mean_token_logprob: Optional[float]
    min_token_logprob: Optional[float]
    sequence_logprob: Optional[float]
    token_logprobs: List[Dict[str, Any]]
    cell_precision: float
    cell_recall: float
    cell_f1: float
    exact_match: bool
    partial_match: bool
    error: Optional[str]
    timestamp: str


def _json_schema() -> Dict[str, Any]:
    return AttributionPrediction.model_json_schema()


def _append_jsonl(path: Path, records: List[VLLMRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _token_logprob_summary(choice: Any, max_tokens_to_store: int) -> tuple[Optional[float], Optional[float], Optional[float], List[Dict[str, Any]]]:
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None, None, None, []

    logprobs = []
    token_rows: List[Dict[str, Any]] = []
    for item in content:
        lp = getattr(item, "logprob", None)
        if lp is None:
            continue
        logprobs.append(float(lp))
        if len(token_rows) < max_tokens_to_store:
            token_rows.append(
                {
                    "token": getattr(item, "token", None),
                    "logprob": float(lp),
                    "prob": math.exp(float(lp)) if float(lp) > -745 else 0.0,
                    "bytes": getattr(item, "bytes", None),
                }
            )

    if not logprobs:
        return None, None, None, token_rows
    return (
        sum(logprobs) / len(logprobs),
        min(logprobs),
        sum(logprobs),
        token_rows,
    )


def _parse_structured(raw_output: str) -> tuple[List[str], bool]:
    try:
        parsed = AttributionPrediction.model_validate_json(raw_output)
        cells = [cell.upper().lstrip("=").strip() for cell in parsed.cells if cell.strip()]
        return cells, True
    except (ValidationError, ValueError):
        return parse_model_output(raw_output), False


async def _run_one(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    sample: Any,
    prompt: str,
    ground_truth_cells: List[str],
    semaphore: asyncio.Semaphore,
) -> VLLMRecord:
    async with semaphore:
        start = time.perf_counter()
        error = None
        raw_output = ""
        input_tokens = None
        output_tokens = None
        mean_lp = None
        min_lp = None
        seq_lp = None
        token_logprobs: List[Dict[str, Any]] = []

        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON matching the requested schema. "
                            "Use Excel-style cell coordinates without extra prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                logprobs=args.logprobs,
                top_logprobs=args.top_logprobs if args.logprobs else None,
                extra_body={"guided_json": _json_schema()} if args.guided_json else None,
            )
            choice = response.choices[0]
            raw_output = _extract_text(choice.message.content).strip()
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            mean_lp, min_lp, seq_lp, token_logprobs = _token_logprob_summary(
                choice,
                args.max_logprob_tokens,
            )
        except Exception as exc:
            error = repr(exc)
            logger.exception("Inference failed for sample %s", sample.id)

        elapsed_ms = (time.perf_counter() - start) * 1000
        predicted_cells, parsed_structured = _parse_structured(raw_output)
        metrics = evaluate_single_prediction(predicted_cells, ground_truth_cells)

        return VLLMRecord(
            sample_id=sample.id,
            model_name=args.model,
            representation=args.representation,
            strategy=args.strategy,
            question=sample.question,
            answer=sample.answer,
            ground_truth_cells=ground_truth_cells,
            predicted_cells=predicted_cells,
            parsed_structured=parsed_structured,
            raw_output=raw_output,
            prompt=prompt,
            inference_time_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mean_token_logprob=mean_lp,
            min_token_logprob=min_lp,
            sequence_logprob=seq_lp,
            token_logprobs=token_logprobs,
            cell_precision=metrics.cell_precision,
            cell_recall=metrics.cell_recall,
            cell_f1=metrics.cell_f1,
            exact_match=metrics.exact_match,
            partial_match=metrics.partial_match,
            error=error,
            timestamp=datetime.now().isoformat(),
        )


async def run(args: argparse.Namespace) -> Path:
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    dataset = VisualCiteDataset(
        jsonl_path=args.jsonl_path,
        split=args.split,
        max_samples=args.max_samples,
        single_cell_only=args.single_cell_only,
    )
    dataset.load()

    representation = DataRepresentation(args.representation)
    strategy = PromptStrategy(args.strategy)
    prompt_builder = PromptBuilder(jsonl_path=args.jsonl_path, num_examples=args.num_examples)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    semaphore = asyncio.Semaphore(args.concurrency)
    for sample in dataset:
        table_content = dataset.get_table_representation(sample, representation.value)
        if not isinstance(table_content, str):
            raise ValueError("vllm_inference.py currently supports text representations only: json, markdown")
        prompt, _ = prompt_builder.build_prompt(sample, strategy, representation, table_content)
        ground_truth_cells = dataset.get_ground_truth_cells(sample)
        tasks.append(_run_one(client, args, sample, prompt, ground_truth_cells, semaphore))

    pending: List[VLLMRecord] = []
    completed = 0
    with tqdm(total=len(tasks), desc="vLLM inference") as pbar:
        for coro in asyncio.as_completed(tasks):
            pending.append(await coro)
            completed += 1
            pbar.update(1)
            if len(pending) >= args.write_every or completed == len(tasks):
                _append_jsonl(output_path, pending)
                pending.clear()
            if completed % args.log_every == 0 or completed == len(tasks):
                logger.info("Completed %s/%s samples; latest output: %s", completed, len(tasks), output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async VisualCite inference via vLLM OpenAI server")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True, help="Served model name, usually --served-model-name from vLLM")
    parser.add_argument("--jsonl-path", default="../visualcite.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--single-cell-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--representation", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--strategy", choices=[s.value for s in PromptStrategy], default=PromptStrategy.ZERO_SHOT.value)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--write-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-logprob-tokens", type=int, default=128)
    parser.add_argument("--guided-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-path", default="vllm_results/predictions.jsonl")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    output_path = asyncio.run(run(args))
    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
