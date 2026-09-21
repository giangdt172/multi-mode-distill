#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
hf download VoCuc/UltraInteract-Infer \
    Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
    --repo-type dataset --local-dir ./data/
