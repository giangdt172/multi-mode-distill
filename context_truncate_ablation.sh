#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_PATH
cd "$BASE_PATH"

export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
export CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"
export DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/$CKPT}"
export CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Keep adaptive OFF + SELF + ON unchanged. Only the SELF context truncation
# strategy is ablated: full context versus removing the final 25% of steps.
CONTEXT_TRUNCATE_RATIOS=(0.00 0.25)
ADAPTIVE_MODE_SET=all
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/context_truncate_ablation}"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -s "$DATA_DIR/train.jsonl" && ( -s "$DATA_DIR/valid.jsonl" || -s "$DATA_DIR/dev.jsonl" ) ]] || {
        printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
        exit 1
    }
    VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
    source "$VENV_PATH/bin/activate"
fi

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

for RATIO in "${CONTEXT_TRUNCATE_RATIOS[@]}"; do
    RATIO_TAG="${RATIO//./p}"
    RUN_SAVE_PATH="$RESULTS_ROOT/drop_${RATIO_TAG}"
    printf '\n[drop=%s 1/2] Train with a fixed SELF context truncation ratio\n' "$RATIO"
    : > "$CHECKPOINT_FILE"
    ADAPTIVE_MODE_SET="$ADAPTIVE_MODE_SET" \
        SAVE_PATH="$RUN_SAVE_PATH" FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
        bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
        "$@" --self-distill-context-fixed-drop-ratio "$RATIO"

    if [[ "${DRY_RUN:-0}" == 1 || "${RUN_EVAL:-1}" == 0 ]]; then
        continue
    fi

    LORA_PATH="$(cat "$CHECKPOINT_FILE")"
    [[ -f "$LORA_PATH/adapter_config.json" ]] || {
        printf 'Final LoRA checkpoint missing for ratio %s: %s\n' \
            "$RATIO" "$LORA_PATH" >&2
        exit 1
    }
    printf '\n[drop=%s 2/2] Evaluate checkpoint: %s\n' "$RATIO" "$LORA_PATH"
    LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
        SAVE_PATH="$(dirname -- "$LORA_PATH")" \
        EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
        bash scripts/eval/eval.sh run
done
