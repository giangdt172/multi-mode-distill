#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_PATH"
export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=. python ./tools/process_data_ultraInteract.py \
    --data-dir ./data/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
    --processed-data-dir ./processed_data/ultraInteract \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --data-process-workers 32 \
    --max-length 2048 \
    --max-prompt-length 512 \
    --dev-num 200 \
    --model-type qwen "$@"
