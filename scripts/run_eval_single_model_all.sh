mkdir -p vllm_results/eval

MODEL=gemma3-4b-it

for FILE in vllm_results/$MODEL/*.jsonl; do
    PYTHONPATH=src python src/evaluate_predictions.py \
        "$FILE" \
        --output-dir vllm_results/$MODEL/eval \
        --ece-bins 10
done