#!/usr/bin/env python3
"""Self-contained asynchronous VisualCite inference through a vLLM OpenAI server."""

import argparse
import ast
import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from tqdm import tqdm


logger = logging.getLogger(__name__)
CELL_RE = re.compile(r"([A-Za-z]+[0-9]+)(?::([A-Za-z]+[0-9]+))?")
IMAGE_REPRESENTATIONS = (
    "image",
    "image_arial",
    "image_times_new_roman",
    "image_red",
    "image_blue",
    "image_green",
)
STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought")

ATTRIBUTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^[A-Za-z]+[1-9][0-9]*(?::[A-Za-z]+[1-9][0-9]*)?$",
            },
            "description": "Excel-style cells supporting the answer, such as E7.",
        },
        "rationale": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        },
    },
    "required": ["cells"],
    "additionalProperties": False,
}


@dataclass
class Sample:
    sample_id: str
    source_sample_id: str
    split: str
    question: str
    answer: str
    answer_formulas: List[str]
    highlighted_cells: List[List[int]]
    table_text: Optional[str]
    representation: str
    image_base64: Optional[str]


@dataclass
class SampleMetrics:
    precision: float
    recall: float
    f1: float
    exact_match: bool
    partial_match: bool


@dataclass
class VLLMRecord:
    source_sample_id: str
    sample_index: int
    temperature: float
    sample_id: str
    split: str
    model_name: str
    representation: str
    strategy: str
    question: str
    answer: str
    ground_truth_cells: List[str]
    predicted_cells: List[str]
    parsed_structured: bool
    guided_json: bool
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


def _column_name(index: int) -> str:
    if index < 0:
        raise ValueError(f"Column index must be non-negative, got {index}")
    result = ""
    while index >= 0:
        result = chr(ord("A") + index % 26) + result
        index = index // 26 - 1
    return result


def _json_with_coordinates(table_json: Dict[str, Any]) -> str:
    """Serialize the JSON table as an explicit Excel-coordinate cell map."""
    texts = table_json.get("texts", [])
    cells: Dict[str, Any] = {}
    if isinstance(texts, list):
        for row_index, row in enumerate(texts):
            if not isinstance(row, list):
                continue
            for column_index, value in enumerate(row):
                cells[f"{_column_name(column_index)}{row_index + 1}"] = value

    payload = {
        "title": table_json.get("title"),
        "coordinate_system": (
            "Each key in cells is the canonical Excel coordinate: column letters "
            "followed by the 1-based row number. Return these keys verbatim."
        ),
        "cells": cells,
        "header_rows": table_json.get("top_header_rows_num"),
        "header_columns": table_json.get("left_header_columns_num"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _cell_indices(cell: str) -> Tuple[int, int]:
    match = re.fullmatch(r"([A-Za-z]+)([0-9]+)", cell.strip())
    if not match:
        raise ValueError(f"Invalid cell coordinate: {cell}")
    column_text, row_text = match.groups()
    column = 0
    for character in column_text.upper():
        column = column * 26 + ord(character) - ord("A") + 1
    return int(row_text) - 1, column - 1


def _expand_range(start: str, end: str) -> List[str]:
    start_row, start_column = _cell_indices(start)
    end_row, end_column = _cell_indices(end)
    if end_row < start_row or end_column < start_column:
        return []
    return [
        f"{_column_name(column)}{row + 1}"
        for row in range(start_row, end_row + 1)
        for column in range(start_column, end_column + 1)
    ]


def _deduplicate(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip().lstrip("=").strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _extract_formula_cells(formulas: Sequence[str]) -> List[str]:
    cells: List[str] = []
    for formula in formulas:
        for start, end in CELL_RE.findall(str(formula)):
            if end:
                cells.extend(_expand_range(start.upper(), end.upper()))
            else:
                cells.append(start.upper())
    return _deduplicate(cells)


def _highlighted_cells_to_coordinates(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cells = []
    for item in value:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
            and item[0] >= 0
            and item[1] >= 0
        ):
            cells.append(f"{_column_name(item[1])}{item[0] + 1}")
    return _deduplicate(cells)


def _parse_formulas(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    return []


def _ground_truth_cells(sample: Sample) -> List[str]:
    formula_cells = _extract_formula_cells(sample.answer_formulas)
    return formula_cells or _highlighted_cells_to_coordinates(sample.highlighted_cells)


def _image_style(representation: str) -> str:
    if representation in ("image", "image_arial"):
        return "arial"
    return representation.removeprefix("image_")


def _sample_from_row(row: Dict[str, Any], representation: str) -> Sample:
    is_image = representation.startswith("image")
    table_text: Optional[str] = None
    image_base64: Optional[str] = None
    if is_image:
        images = row.get("table_images") or {}
        if not isinstance(images, dict):
            raise ValueError("table_images must be an object")
        style = _image_style(representation)
        selected = images.get(style) or images.get("arial")
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"No usable '{style}' or 'arial' table image")
        image_base64 = selected.strip()
    elif representation == "markdown":
        value = row.get("table_md")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("table_md is missing or empty")
        table_text = value
    elif representation == "json":
        value = row.get("table_json")
        if not isinstance(value, dict):
            raise ValueError("table_json is missing or is not an object")
        table_text = _json_with_coordinates(value)
    else:
        raise ValueError(f"Unknown representation: {representation}")

    highlighted = row.get("highlighted_cells") or []
    return Sample(
        sample_id=str(row.get("id", "")).strip(),
        split=str(row.get("split", "")).strip(),
        source_sample_id=str(row.get("id", "")).strip(),
        question=str(row.get("question", "")),
        representation=representation,
        answer=str(row.get("answer", "")),
        answer_formulas=_parse_formulas(row.get("answer_formulas", [])),
        highlighted_cells=highlighted if isinstance(highlighted, list) else [],
        table_text=table_text,
        image_base64=image_base64,
    )


def _load_samples(
    path: Path,
    split: str,
    representation: str,
    max_samples: Optional[int],
    single_cell_only: bool,
) -> List[Sample]:
    samples: List[Sample] = []
    representations = IMAGE_REPRESENTATIONS[1:] if representation == "image" else (representation,)
    target_count = max_samples * len(representations) if max_samples is not None else None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %s: %s", line_number, exc)
                continue
            if not isinstance(row, dict) or str(row.get("split", "")).lower() != split.lower():
                continue
            for variant in representations:
                try:
                    sample = _sample_from_row(row, variant)
                    if not sample.sample_id:
                        raise ValueError("sample id is empty")
                    truth = _ground_truth_cells(sample)
                    if not truth:
                        raise ValueError("ground truth is empty")
                    if single_cell_only and len(truth) != 1:
                        continue
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping sample on line %s: %s", line_number, exc)
                    continue
                if representation == "image":
                    sample.sample_id = f"{sample.sample_id}__{variant}"
                samples.append(sample)
                if target_count is not None and len(samples) >= target_count:
                    break
            if target_count is not None and len(samples) >= target_count:
                break
    logger.info("Loaded %s samples for split=%s representation=%s", len(samples), split, representation)
    return samples


def _load_few_shot_example(
    path: Path, target_split: str, representation: str, num_examples: int
) -> Optional[Sample]:
    preferred_splits = ("validation", "dev", "train", "test")
    for split in preferred_splits:
        if split == target_split:
            continue
        examples = _load_samples(path, split, representation, num_examples, True)
        if examples:
            return examples[min(1, len(examples) - 1)]
    return None


def _base_instruction(question: str, answer: str) -> str:
    return f"""You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

QUESTION: {question}
ANSWER: {answer}

TASK: Identify the cell coordinate(s) that contain or directly support this answer. Use Excel-style coordinates where columns are letters (A, B, C, ...) and rows are numbers (1, 2, 3, ...).

RESPONSE FORMAT: Return ONLY the cell coordinates in Excel formula format. Examples:
- Single cell: "=E7" or "=B3"
- Multiple cells: "=A2" or list them separately: "=A2, =B2, =C2"
- If the answer involves a formula (sum, average, etc.), you may use: "SUM(C3:C10)" or "=A1+B2"

IMPORTANT: Do NOT repeat the question, table, or instructions. Output ONLY the cell coordinates.

ATTRIBUTED CELLS:"""


def _response_format_instruction(guided_json: bool, allow_rationale: bool = False) -> str:
    if guided_json:
        rationale_text = (
            ' Include a concise "rationale" string before the final cells.'
            if allow_rationale
            else ' Use null for "rationale".'
        )
        return (
            'Return ONLY a JSON object matching this shape: '
            '{"cells":["E7"],"rationale":null}. '
            'Put Excel-style cell coordinates in "cells" without a leading "=".'
            f"{rationale_text}"
        )
    return """Return ONLY the cell coordinates in Excel formula format. Examples:
- Single cell: "=E7" or "=B3"
- Multiple cells: "=A2" or list them separately: "=A2, =B2, =C2"
- If the answer involves a formula (sum, average, etc.), you may use: "SUM(C3:C10)" or "=A1+B2"."""


def _build_prompt(
    sample: Sample, strategy: str, example: Optional[Sample], guided_json: bool
) -> str:
    table = sample.table_text if sample.table_text is not None else "[TABLE IMAGE PROVIDED]"
    coordinate_instruction = (
        'For this JSON table, the keys inside the top-level "cells" object '
        "(for example A1 or B3) are the canonical coordinates. Return those keys; "
        "do not return numeric [row, column] indices."
        if sample.representation == "json"
        else "Use the displayed row numbers and column letters as coordinates."
    )
    if strategy == "zero_shot":
        template = """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

TASK: Identify the cell coordinate(s) that contain or directly support this answer. Use Excel-style coordinates where columns are letters (A, B, C, ...) and rows are numbers (1, 2, 3, ...).
COORDINATE SYSTEM: {coordinate_instruction}

RESPONSE FORMAT: {response_format}

IMPORTANT: Do NOT repeat the question, table, or instructions. Output ONLY the requested response.

ATTRIBUTED CELLS:"""
        return template.format(
            table=table,
            question=sample.question,
            answer=sample.answer,
            coordinate_instruction=coordinate_instruction,
            response_format=_response_format_instruction(guided_json),
        )
    if strategy == "chain_of_thought":
        template = """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

Let's think step by step:

1. First, understand what the question is asking for.
2. Then, locate where the answer "{answer}" appears or can be derived from in the table.
3. Identify the specific cell coordinate(s) using Excel-style notation (columns as letters, rows as numbers 1, 2, 3...).
   {coordinate_instruction}
4. If the answer is computed from multiple cells (e.g., a sum), include the source cells or range.
5. Use this response format: {response_format}

IMPORTANT: Do NOT repeat the question or table in your reasoning.

REASONING:
"""
        return template.format(
            table=table,
            question=sample.question,
            answer=sample.answer,
            coordinate_instruction=coordinate_instruction,
            response_format=_response_format_instruction(guided_json, allow_rationale=True),
        )
    if strategy != "few_shot":
        raise ValueError(f"Unknown strategy: {strategy}")

    example_table = example.table_text if example is not None else "| A | B |\\n| 1 | 2 |"
    example_question = example.question if example is not None else "What is in cell A1?"
    example_answer = example.answer if example is not None else "1"
    example_cells = _ground_truth_cells(example) if example is not None else ["A1"]
    example_response = (
        json.dumps({"cells": example_cells, "rationale": None})
        if guided_json
        else ", ".join(f"={cell}" for cell in example_cells)
    )
    template = """You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

Here is an example:

EXAMPLE:
TABLE:
{example1_table}
QUESTION: {example1_question}
ANSWER: {example1_answer}
ATTRIBUTED CELLS: {example1_response}

Now analyze this table:

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

COORDINATE SYSTEM: {coordinate_instruction}
RESPONSE FORMAT: {response_format}

IMPORTANT: Do NOT repeat the example, question, table, or instructions.

ATTRIBUTED CELLS:"""
    return template.format(
        example1_table=example_table,
        example1_question=example_question,
        example1_answer=example_answer,
        example1_response=example_response,
        table=table,
        question=sample.question,
        answer=sample.answer,
        coordinate_instruction=coordinate_instruction,
        response_format=_response_format_instruction(guided_json),
    )

def _user_content(prompt: str, image_base64: Optional[str]) -> Any:
    if image_base64 is None:
        return prompt
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        },
        {"type": "text", "text": prompt},
    ]


def _parse_model_output(output: str) -> Tuple[List[str], bool]:
    try:
        parsed = json.loads(output)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("cells"), list):
            raise ValueError("JSON output does not contain a cells array")
        if any(not isinstance(cell, str) for cell in parsed["cells"]):
            raise ValueError("cells array contains a non-string value")
        cells: List[str] = []
        for value in parsed["cells"]:
            match = re.fullmatch(
                r"=?\s*([A-Za-z]+[1-9][0-9]*)(?::([A-Za-z]+[1-9][0-9]*))?\s*",
                value,
            )
            if not match:
                raise ValueError(f"invalid cell coordinate: {value}")
            start, end = match.groups()
            if end:
                cells.extend(_expand_range(start.upper(), end.upper()))
            else:
                cells.append(start.upper())
        return _deduplicate(cells), True
    except (json.JSONDecodeError, ValueError, TypeError):
        cells: List[str] = []
        for start, end in CELL_RE.findall(output):
            if end:
                cells.extend(_expand_range(start.upper(), end.upper()))
            else:
                cells.append(start.upper())
        return _deduplicate(cells), False


def _evaluate(predicted: Sequence[str], truth: Sequence[str]) -> SampleMetrics:
    predicted_set = set(_deduplicate(predicted))
    truth_set = set(_deduplicate(truth))
    if not predicted_set and not truth_set:
        precision = recall = f1 = 1.0
    elif not predicted_set or not truth_set:
        precision = recall = f1 = 0.0
    else:
        true_positives = len(predicted_set & truth_set)
        precision = true_positives / len(predicted_set)
        recall = true_positives / len(truth_set)
        f1 = 2.0 * precision * recall / (precision + recall) if true_positives else 0.0
    return SampleMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=predicted_set == truth_set,
        partial_match=bool(predicted_set & truth_set),
    )


def _append_jsonl(path: Path, records: Sequence[VLLMRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


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


def _token_logprob_summary(
    choice: Any, max_tokens_to_store: int
) -> Tuple[Optional[float], Optional[float], Optional[float], List[Dict[str, Any]]]:
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None, None, None, []

    logprobs: List[float] = []
    token_rows: List[Dict[str, Any]] = []
    for item in content:
        value = getattr(item, "logprob", None)
        if value is None:
            continue
        logprob = float(value)
        logprobs.append(logprob)
        if len(token_rows) < max_tokens_to_store:
            token_rows.append(
                {
                    "token": getattr(item, "token", None),
                    "logprob": logprob,
                    "prob": math.exp(logprob) if logprob > -745 else 0.0,
                    "bytes": getattr(item, "bytes", None),
                }
            )
    if not logprobs:
        return None, None, None, token_rows
    return sum(logprobs) / len(logprobs), min(logprobs), sum(logprobs), token_rows


async def _run_one(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    sample: Sample,
    prompt: str,
    sample_index: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> VLLMRecord:
    async with semaphore:
        started = time.perf_counter()
        error: Optional[str] = None
        raw_output = ""
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        mean_logprob: Optional[float] = None
        min_logprob: Optional[float] = None
        sequence_logprob: Optional[float] = None
        token_logprobs: List[Dict[str, Any]] = []
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": _user_content(prompt, sample.image_base64)}],
                temperature=temperature,
                max_tokens=args.max_tokens,
                logprobs=args.logprobs,
                top_logprobs=args.top_logprobs if args.logprobs else None,
                extra_body={"guided_json": ATTRIBUTION_SCHEMA} if args.guided_json else None,
            )
            choice = response.choices[0]
            raw_output = _extract_text(choice.message.content).strip()
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            mean_logprob, min_logprob, sequence_logprob, token_logprobs = (
                _token_logprob_summary(choice, args.max_logprob_tokens)
            )
        except Exception as exc:
            error = repr(exc)
            logger.exception("Inference failed for sample %s", sample.sample_id)

        predicted_cells, parsed_structured = _parse_model_output(raw_output)
        truth = _ground_truth_cells(sample)
        metrics = _evaluate(predicted_cells, truth)
        return VLLMRecord(
            sample_id=f"{sample.sample_id}__draw_{sample_index}",
            source_sample_id=sample.source_sample_id,
            sample_index=sample_index,
            temperature=temperature,
            split=sample.split,
            model_name=args.model,
            representation=sample.representation,
            strategy=args.strategy,
            question=sample.question,
            answer=sample.answer,
            ground_truth_cells=truth,
            predicted_cells=predicted_cells,
            parsed_structured=parsed_structured,
            guided_json=args.guided_json,
            raw_output=raw_output,
            prompt=prompt,
            inference_time_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mean_token_logprob=mean_logprob,
            min_token_logprob=min_logprob,
            sequence_logprob=sequence_logprob,
            token_logprobs=token_logprobs,
            cell_precision=metrics.precision,
            cell_recall=metrics.recall,
            cell_f1=metrics.f1,
            exact_match=metrics.exact_match,
            partial_match=metrics.partial_match,
            error=error,
            timestamp=datetime.now().isoformat(),
        )


def _temperature_values(args: argparse.Namespace) -> List[float]:
    if args.temperatures:
        values = []
        for raw_value in args.temperatures.split(","):
            try:
                value = float(raw_value.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid temperature: {raw_value}") from exc
            if value < 0.0:
                raise ValueError("Temperatures must be non-negative")
            values.append(value)
        if not values:
            raise ValueError("--temperatures cannot be empty")
        return values
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.temperature < 0.0:
        raise ValueError("Temperature must be non-negative")
    return [args.temperature] * args.num_samples


async def run(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.jsonl_path)
    samples = _load_samples(
        dataset_path,
        args.split,
        args.representation,
        args.max_samples,
        args.single_cell_only,
    )
    example = (
        _load_few_shot_example(
            dataset_path, args.split, args.representation, args.num_examples
        )
        if args.strategy == "few_shot"
        else None
    )
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    temperature_values = _temperature_values(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    def request_coroutines() -> Iterable[Any]:
        for sample in samples:
            prompt = _build_prompt(sample, args.strategy, example, args.guided_json)
            for sample_index, temperature in enumerate(temperature_values):
                yield _run_one(
                    client, args, sample, prompt, sample_index, temperature, semaphore
                )

    pending: List[VLLMRecord] = []
    completed = 0
    total = len(samples) * len(temperature_values)
    coroutine_iter = iter(request_coroutines())
    active: set[asyncio.Task[VLLMRecord]] = set()

    def fill_active() -> None:
        while len(active) < args.concurrency:
            try:
                active.add(asyncio.create_task(next(coroutine_iter)))
            except StopIteration:
                break

    with tqdm(total=total, desc="vLLM inference") as progress:
        fill_active()
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            fill_active()
            for task in done:
                pending.append(await task)
                completed += 1
                progress.update(1)
                if len(pending) >= args.write_every or completed == total:
                    _append_jsonl(output_path, pending)
                    pending.clear()
                if completed % args.log_every == 0 or completed == total:
                    logger.info("Completed %s/%s; output=%s", completed, total, output_path)
    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained VisualCite inference through a vLLM OpenAI server"
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True, help="Name supplied to vLLM with --served-model-name")
    parser.add_argument("--jsonl-path", default="visualcite.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--single-cell-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--representation",
        choices=("json", "markdown", *IMAGE_REPRESENTATIONS),
        default="markdown",
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default="zero_shot")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1, help="Repeated draws when --temperatures is omitted")
    parser.add_argument("--temperatures", default=None, help="Comma-separated temperatures for repeated UQ draws")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--write-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--max-logprob-tokens", type=int, default=128)
    parser.add_argument("--guided-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-path", default="vllm_results/predictions.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    for name in ("max_samples", "num_examples", "num_samples", "concurrency", "write_every", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.max_logprob_tokens < 0:
        parser.error("--max-logprob-tokens must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    output_path = asyncio.run(run(args))
    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
