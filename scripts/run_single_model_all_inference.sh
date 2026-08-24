#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-vitab-model}"
MODEL_LABEL="${MODEL_LABEL:-gemma3_4b_it}"
BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
JSONL_PATH="${JSONL_PATH:-visualcite.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-vllm_results}"
CONCURRENCY="${CONCURRENCY:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-999999999}"
FORCE="${FORCE:-0}"

# SPLITS=(dev)
SPLITS=(train validation dev test)
STRATEGIES=(zero_shot few_shot chain_of_thought)
MODALITIES=(markdown image json)
# MODALITIES=(json)

mkdir -p "$OUTPUT_DIR"

for MODALITY in "${MODALITIES[@]}"; do
    case "$MODALITY" in
        markdown|image|json) ;;
        *)
            echo "Unsupported modality for src/vllm_inference.py: $MODALITY" >&2
            exit 2
            ;;
    esac

    for STRATEGY in "${STRATEGIES[@]}"; do
        OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_LABEL}_${MODALITY}_${STRATEGY}_all_splits.jsonl"
        PARTIAL_PATH="${OUTPUT_PATH}.partial"

        if [[ -e "$OUTPUT_PATH" && "$FORCE" != "1" ]]; then
            echo "Output already exists: $OUTPUT_PATH" >&2
            echo "Set FORCE=1 to replace existing combination files." >&2
            exit 1
        fi

        # Inference appends records, so each combination starts with a fresh partial file.
        : > "$PARTIAL_PATH"
        echo "Running modality=$MODALITY strategy=$STRATEGY -> $OUTPUT_PATH"

        for SPLIT in "${SPLITS[@]}"; do
            echo "  split=$SPLIT"
            OPENAI_BASE_URL="$BASE_URL" \
            OPENAI_API_KEY="$API_KEY" \
            python src/vllm_inference.py \
                --model "$MODEL_NAME" \
                --jsonl-path "$JSONL_PATH" \
                --split "$SPLIT" \
                --representation "$MODALITY" \
                --strategy "$STRATEGY" \
                --max-samples "$MAX_SAMPLES" \
                --concurrency "$CONCURRENCY" \
                --output-path "$PARTIAL_PATH" \
                --temperatures 0.3,0.5,0.7,0.9 
        done

        mv -f "$PARTIAL_PATH" "$OUTPUT_PATH"
        echo "Completed $OUTPUT_PATH"
    done
done
