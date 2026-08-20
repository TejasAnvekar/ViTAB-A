# ViTAB

Unified benchmark framework for VisualCite table-cell attribution across multiple vision-language models.

ViTAB evaluates how well models identify the supporting table cell(s) for a question-answer pair, with support for:
- Multiple model families: Gemma, InternVL 3.5, Qwen3-VL, Molmo2
- Multiple table representations: JSON, Markdown, and styled images
- Multiple prompting strategies: zero-shot, few-shot, chain-of-thought
- Confidence analysis: internal confidence, verbalized certainty, alignment metrics
- Uncertainty quantification: split conformal prediction (LAC + APS)

## Project Structure

This repository is consolidated into a single source directory:

```text
ViTAB-A/
├── README.md
└── src/
		├── benchmark_runner.py
		├── confidence_benchmark_runner.py
		├── model_handler.py
		├── data_loader.py
		├── prompt_builder.py
		├── metrics.py
		├── uncertainty_quantification.py
		├── test_benchmark.py
		└── requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for model inference
- VisualCite dataset file in JSONL format

Install dependencies:

```bash
cd src
pip install -r requirements.txt
```

## Dataset Setup

By default, the runner expects:

```text
../visualcite.jsonl
```

relative to `src/`.

So with the default configuration, place your dataset at:

```text
ViTAB-A/visualcite.jsonl
```

You can override this with `--jsonl-path`.

## Quick Start

Run a small smoke test from `src/`:

```bash
cd src
python benchmark_runner.py \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

## Running `benchmark_runner.py` Properly

`benchmark_runner.py` uses relative defaults (especially for `--jsonl-path`), so run it in one of these two ways:

### Option A (recommended): run from `src/`

```bash
cd src
python benchmark_runner.py \
	--jsonl-path ../visualcite.jsonl \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

### Option B: run from repository root

```bash
cd ..  # if you are currently inside src
python src/benchmark_runner.py \
	--jsonl-path ./visualcite.jsonl \
	--models Qwen/Qwen3-VL-2B-Instruct \
	--representations markdown \
	--strategies zero_shot \
	--max-samples 5
```

Important:
- If you run from root, do **not** rely on the default `--jsonl-path`.
- Always set `--jsonl-path` explicitly when your working directory is not `src/`.

Run a larger benchmark:

```bash
cd src
python benchmark_runner.py \
	--models Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct \
	--representations json markdown image_arial \
	--strategies zero_shot few_shot chain_of_thought \
	--max-samples 200 \
	--split dev
```

## Core CLI Options

Common options:
- `--models`: one or more HF model IDs
- `--representations`: `json`, `markdown`, `image_arial`, `image_times_new_roman`, `image_red`, `image_blue`, `image_green`
- `--strategies`: `zero_shot`, `few_shot`, `chain_of_thought`
- `--jsonl-path`: path to dataset JSONL
- `--max-samples`: maximum samples to process
- `--split`: `train`, `validation`, `dev`, `test`
- `--single-cell-only`: filter to samples with exactly one ground-truth cell

Runtime and resume:
- `--output-dir`: output directory (default `benchmark_results`)
- `--checkpoint-dir`: checkpoint directory (default `checkpoints`)
- `--no-resume`: disable checkpoint resume
- `--clear-checkpoints`: remove previous checkpoints before run

Model/runtime settings:
- `--device`: `cuda` or `cpu`
- `--dtype`: `float16`, `bfloat16`, `float32`
- `--no-flash-attention`

Confidence/UQ options:
- `--extract-internal-confidence`
- `--extract-verbalized-certainty`
- `--cqp-template`, `--cqp-max-tokens`, `--cqp-temperature`
- `--enable-conformal-uq`
- `--conformal-calibration-ratio`, `--conformal-alpha`, `--conformal-seed`

## Testing

Run lightweight validation tests:

```bash
cd src
python test_benchmark.py --test all
```

Or specific test groups:

```bash
python test_benchmark.py --test metrics
python test_benchmark.py --test prompts
python test_benchmark.py --test parsing
python test_benchmark.py --test checkpoint
```

## Outputs

By default, outputs are written under `src/benchmark_results/`:
- instance logs (per sample)
- verbalized certainty logs
- aggregated summaries (JSON)
- CSV exports
- markdown report
- uncertainty summaries (if enabled)

Checkpoints are stored under `src/checkpoints/`.

## Notes

- Running from `src/` is recommended because default paths are defined relative to that directory.
- If few-shot validation examples are unavailable at the dataset path, the prompt builder falls back to placeholder examples.
- Large models may require significant GPU memory and can offload to CPU/disk automatically.
