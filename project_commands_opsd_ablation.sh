#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_PATH
cd "$BASE_PATH"

export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
export CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"
export DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/$CKPT}"
export CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-8}" GRAD_ACC="${GRAD_ACC:-4}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export LR="${LR:-5e-6}" LORA_R="${LORA_R:-64}"
OPSD_TOP_K="${OPSD_TOP_K:-20}" OPSD_TOP_P="${OPSD_TOP_P:-0.95}"
OPSD_TEMPERATURE="${OPSD_TEMPERATURE:-1.1}"
OPSD_WEIGHT_DECAY="${OPSD_WEIGHT_DECAY:-0.0}" OPSD_CLIP_GRAD="${OPSD_CLIP_GRAD:-0.1}"
OPSD_SAVE_INTERVAL="${OPSD_SAVE_INTERVAL:-20}"
export EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-$LORA_R}"

export KD_LOSS=fkl KD_RATIO=1.0 GEOMETRY=0 DISTILL_TEMPERATURE=1.0
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export MAX_LENGTH="${MAX_LENGTH:-1536}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-4096}"
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((T_MAX_PROMPT_LENGTH + MAX_LENGTH - MAX_PROMPT_LENGTH))}"
export OPSD_TOKEN_CLIP="${OPSD_TOKEN_CLIP:-0.05}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
export SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-opsd/fkl_fullvocab_clip${OPSD_TOKEN_CLIP}_lr${LR}_r${LORA_R}_bs${BATCH_SIZE}_ga${GRAD_ACC}_seed${SEED:-10}}"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -s "$DATA_DIR/train.jsonl" && ( -s "$DATA_DIR/valid.jsonl" || -s "$DATA_DIR/dev.jsonl" ) ]] || {
        printf 'OPSD requires processed train and valid/dev JSONL files in %s\n' "$DATA_DIR" >&2
        exit 1
    }
    VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
    source "$VENV_PATH/bin/activate"
fi

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_single_mode_qwen2.5_14b_to_1.5b.sh opsd \
    --total-iters "${OPSD_TOTAL_ITERS:-100}" \
    --top-k "$OPSD_TOP_K" --top-p "$OPSD_TOP_P" --temperature "$OPSD_TEMPERATURE" \
    --weight-decay "$OPSD_WEIGHT_DECAY" --clip-grad "$OPSD_CLIP_GRAD" \
    --save-interval "$OPSD_SAVE_INTERVAL" "$@"

if [[ "${DRY_RUN:-0}" == 1 || "${RUN_EVAL:-1}" == 0 ]]; then
    exit 0
fi

LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || {
    printf 'Final OPSD LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
}
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
    LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    bash scripts/eval/eval.sh run
