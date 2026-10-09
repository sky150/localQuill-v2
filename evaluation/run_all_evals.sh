#!/bin/bash

MODELS=(
    "mxbai-embed-large:latest"
    "dengcao/Qwen3-Embedding-0.6B:Q8_0"
    "embeddinggemma:latest"
    "nomic-embed-text:latest"
    "locusai/all-minilm-l6-v2:latest"
)

SLEEP_TIME=20

echo "=== Starting automated retrieval evaluation for ${#MODELS[@]} models ==="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================================="
    echo "▶ Evaluating Model: $MODEL"
    echo "=========================================================="

    echo "Waking up $MODEL..."
    curl -s -X POST http://localhost:11434/api/embeddings -d "{\"model\": \"$MODEL\", \"prompt\": \"test\"}" > /dev/null
    
    echo "Running evaluation..."
    EVAL_MODEL="$MODEL" PYTHONPATH=. caffeinate -d uv run python tests/eval/run_retrieval_eval.py
    
    if [ "$MODEL" != "${MODELS[${#MODELS[@]}-1]}" ]; then
        echo "----------------------------------------------------------"
        echo "Finished $MODEL. Pausing for $SLEEP_TIME seconds to let Ollama breathe..."
        sleep $SLEEP_TIME
    fi
done

echo ""
echo "All model evaluations completed successfully!"
