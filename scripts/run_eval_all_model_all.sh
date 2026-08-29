#!/usr/bin/env bash
set -euo pipefail

# Evaluate every prediction JSONL below a results root. Each summary is written
# to an `eval/` directory beside its source JSONL so model families remain
# separated and repeated runs safely refresh the same files.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${1:-$REPO_ROOT/vllm_results}"
ECE_BINS="${ECE_BINS:-10}"

if [[ ! -d "$RESULTS_ROOT" ]]; then
    echo "Results directory does not exist: $RESULTS_ROOT" >&2
    exit 1
fi

mapfile -d '' FILES < <(
    find "$RESULTS_ROOT" -type f -name '*.jsonl' -not -path '*/eval/*' -print0 | sort -z
)

if (( ${#FILES[@]} == 0 )); then
    echo "No JSONL files found below: $RESULTS_ROOT" >&2
    exit 1
fi

echo "Evaluating ${#FILES[@]} prediction file(s) below $RESULTS_ROOT"
for FILE in "${FILES[@]}"; do
    OUTPUT_DIR="$(dirname -- "$FILE")/eval"
    echo
    echo "[$FILE]"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python "$REPO_ROOT/src/evaluate_predictions.py" \
        "$FILE" \
        --output-dir "$OUTPUT_DIR" \
        --ece-bins "$ECE_BINS"
done

echo
echo "Evaluation complete: ${#FILES[@]} file(s)."
