# ViTAB

Unified benchmark framework for VisualCite table-cell attribution across multiple vision-language models.

ViTAB evaluates how well models identify the supporting table cell(s) for a question-answer pair across model families, table representations, prompting strategies, confidence analysis, and uncertainty quantification.

## Setup

Install the Python dependencies:

```bash
cd src
pip install -r requirements.txt
```

Place the VisualCite dataset at the repository root unless you override `JSONL_PATH`:

```text
ViTAB-A/visualcite.jsonl
```

For gated Hugging Face models, request access on Hugging Face and authenticate before starting vLLM:

```bash
huggingface-cli login
```

or export a token in the environment that starts the model server:

```bash
export HF_TOKEN=hf_your_token_here
```

## Run From Scripts

Run commands from the repository root.

### Serve each configured model and run inference

```bash
bash scripts/run_all_models_serve_then_infer.sh
```

This starts a vLLM OpenAI-compatible server for each configured model, runs inference across the configured splits, strategies, and modalities, then writes results under `vllm_results/`.

Useful overrides:

```bash
MODELS="Qwen3-VL-2B-Instruct" \
SPLITS="dev" \
STRATEGIES="zero_shot" \
MODALITIES="json" \
MAX_SAMPLES=5 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_all_models_serve_then_infer.sh
```

### Run inference against an already running server

```bash
bash scripts/run_single_model_all_inference.sh
```

Useful overrides:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=vitab-model \
MODEL_LABEL=Qwen3-VL-2B-Instruct \
JSONL_PATH=visualcite.jsonl \
OUTPUT_DIR=vllm_results/Qwen3/Qwen3-VL-2B-Instruct \
MAX_SAMPLES=5 \
bash scripts/run_single_model_all_inference.sh
```

### Evaluate generated prediction files

```bash
bash scripts/run_eval_all_model_all.sh
```

Pass a different results root as the first argument if needed:

```bash
bash scripts/run_eval_all_model_all.sh vllm_results
```

Set `ECE_BINS` to change calibration binning:

```bash
ECE_BINS=15 bash scripts/run_eval_all_model_all.sh vllm_results
```

