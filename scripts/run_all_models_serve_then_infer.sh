#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BASE_URL="${OPENAI_BASE_URL:-http://localhost:${PORT}/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vitab-model}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

JSONL_PATH="${JSONL_PATH:-visualcite.jsonl}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-999999999}"
FORCE="${FORCE:-1}"

DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 1}}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"

SPLITS=(${SPLITS:-train validation dev test})
STRATEGIES=(${STRATEGIES:-zero_shot few_shot chain_of_thought})
MODALITIES=(${MODALITIES:-json})

declare -A MODEL_IDS=(
    [InternVL3.5_4b_it]="OpenGVLab/InternVL3_5-4B-Instruct"
    [InternVL3.5_8b_it]="OpenGVLab/InternVL3_5-8B-Instruct"
    [InternVL3.5_14b_it]="OpenGVLab/InternVL3_5-14B-Instruct"
    [InternVL3.5_38b_it]="OpenGVLab/InternVL3_5-38B-Instruct"
    [Qwen3-VL-2B-Instruct]="Qwen/Qwen3-VL-2B-Instruct"
    [Qwen3-VL-4B-Instruct]="Qwen/Qwen3-VL-4B-Instruct"
    [Qwen3-VL-8B-Instruct]="Qwen/Qwen3-VL-8B-Instruct"
    [Qwen3-VL-32B-Instruct]="Qwen/Qwen3-VL-32B-Instruct"
    # [gemma3_4b_it]="google/gemma-3-4b-it"
    [gemma3_12b_it]="google/gemma-3-12b-it"
    [gemma3_27b_it]="google/gemma-3-27b-it"
)

declare -A MODEL_FAMILIES=(
    [InternVL3.5_4b_it]="InternVL3.5"
    [InternVL3.5_8b_it]="InternVL3.5"
    [InternVL3.5_14b_it]="InternVL3.5"
    [InternVL3.5_38b_it]="InternVL3.5"
    [Qwen3-VL-2B-Instruct]="Qwen3"
    [Qwen3-VL-4B-Instruct]="Qwen3"
    [Qwen3-VL-8B-Instruct]="Qwen3"
    [Qwen3-VL-32B-Instruct]="Qwen3"
    # [gemma3_4b_it]="gemma3"
    [gemma3_12b_it]="gemma3"
    [gemma3_27b_it]="gemma3"
)

DEFAULT_MODELS=(
    InternVL3.5_4b_it
    InternVL3.5_8b_it
    InternVL3.5_14b_it
    InternVL3.5_38b_it
    Qwen3-VL-2B-Instruct
    Qwen3-VL-4B-Instruct
    Qwen3-VL-8B-Instruct
    Qwen3-VL-32B-Instruct
    # gemma3_4b_it
    gemma3_12b_it
    gemma3_27b_it
)

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS_TO_RUN <<< "$MODELS"
else
    MODELS_TO_RUN=("${DEFAULT_MODELS[@]}")
fi

SERVER_PID=""
SERVER_LOG=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server pid=$SERVER_PID"
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))

    until curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; do
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM server exited before becoming ready. Last log lines:" >&2
            tail -n 80 "$SERVER_LOG" >&2 || true
            exit 1
        fi

        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for vLLM server at ${BASE_URL}. Last log lines:" >&2
            tail -n 80 "$SERVER_LOG" >&2 || true
            exit 1
        fi

        sleep 5
    done
}

run_inference_for_model() {
    local model_label="$1"
    local family="$2"
    local output_dir="${OUTPUT_DIR:-vllm_results/${family}/${model_label}}"

    mkdir -p "$output_dir"

    for modality in "${MODALITIES[@]}"; do
        case "$modality" in
            markdown|image|json) ;;
            *)
                echo "Unsupported modality for src/vllm_inference.py: $modality" >&2
                exit 2
                ;;
        esac

        for strategy in "${STRATEGIES[@]}"; do
            local output_path="${output_dir}/${model_label}_${modality}_${strategy}_all_splits.jsonl"
            local partial_path="${output_path}.partial"

            if [[ -e "$output_path" && "$FORCE" != "1" ]]; then
                echo "Output already exists: $output_path" >&2
                echo "Set FORCE=1 to replace existing combination files." >&2
                exit 1
            fi

            : > "$partial_path"
            echo "Running model=$model_label modality=$modality strategy=$strategy -> $output_path"

            for split in "${SPLITS[@]}"; do
                echo "  split=$split"
                OPENAI_BASE_URL="$BASE_URL" \
                OPENAI_API_KEY="$API_KEY" \
                python src/vllm_inference.py \
                    --model "$SERVED_MODEL_NAME" \
                    --jsonl-path "$JSONL_PATH" \
                    --split "$split" \
                    --representation "$modality" \
                    --strategy "$strategy" \
                    --max-samples "$MAX_SAMPLES" \
                    --concurrency "$CONCURRENCY" \
                    --output-path "$partial_path" \
                    --temperatures 0.3,0.5,0.7,0.9
            done

            mv -f "$partial_path" "$output_path"
            echo "Completed $output_path"
        done
    done
}

for model_label in "${MODELS_TO_RUN[@]}"; do
    model_id="${MODEL_IDS[$model_label]:-}"
    family="${MODEL_FAMILIES[$model_label]:-}"

    if [[ -z "$model_id" || -z "$family" ]]; then
        echo "Unknown model label: $model_label" >&2
        echo "Known labels: ${DEFAULT_MODELS[*]}" >&2
        exit 2
    fi

    SERVER_LOG="vllm_results/${family}/${model_label}/vllm_server.log"
    mkdir -p "$(dirname "$SERVER_LOG")"

    echo "Starting vLLM server for $model_label ($model_id)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_id" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
        --enable-prefix-caching \
        --trust-remote-code \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"

    wait_for_server
    echo "vLLM server is ready at $BASE_URL"

    run_inference_for_model "$model_label" "$family"
    stop_server
done
