#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_PATH"

ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"

RAW_DATA="$ASSET_ROOT/data/raw/google/gemma-2-9b-it/generated_train.jsonl"
CKPT="$ASSET_ROOT/models/google_gemma-2-2b-it"
TEACHER_CKPT="$ASSET_ROOT/models/google_gemma-2-9b-it"
PROCESSED_DATA_ROOT="$ASSET_ROOT/processed_data/ultraInteract-v2"
DATA_DIR="$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")"
GEMMA_RESULTS_ROOT="${GEMMA_RESULTS_ROOT:-$BASE_PATH/results/gemma-2-2b-it-distill}"
RUN_NAME="${RUN_NAME:-geo${GEOMETRY:-1}_cka${CKA:-0}_menger${MENGER_WEIGHT:-1.0}}"
SAVE_PATH="${SAVE_PATH:-$GEMMA_RESULTS_ROOT/$RUN_NAME}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

uv venv --python 3.11 "$VENV_PATH"
source "$VENV_PATH/bin/activate"
uv pip install -r "$BASE_PATH/multi-mode-distill.txt"

mkdir -p -- "$(dirname -- "$RAW_DATA")" "$CKPT" "$TEACHER_CKPT"
hf download VoCuc/UltraInteract-Infer \
    google/gemma-2-9b-it/generated_train.jsonl \
    --repo-type dataset \
    --local-dir "$ASSET_ROOT/data/raw"
hf download google/gemma-2-2b-it --local-dir "$CKPT"
hf download google/gemma-2-9b-it --local-dir "$TEACHER_CKPT"

PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}" \
python "$BASE_PATH/tools/process_data_ultraInteract.py" \
    --base-path "$BASE_PATH" \
    --data-dir "$RAW_DATA" \
    --processed-data-dir "$PROCESSED_DATA_ROOT" \
    --model-path "$CKPT" \
    --model-type gemma \
    --data-process-workers "${DATA_PROCESS_WORKERS:-32}" \
    --max-length "$MAX_LENGTH" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --dev-num "$DEV_NUM" \
    --seed "$SEED"

KD_LOSS="${KD_LOSS:-sfkl}"
KD_RATIO="${KD_RATIO:-0.5}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
GEOMETRY="${GEOMETRY:-1}"
CKA="${CKA:-0}"
MAG_WEIGHT="${MAG_WEIGHT:-2.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-2.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-1.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
CHECKPOINT_FILE="$(mktemp)"

trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

printf '\n[%s 1/2] Train Gemma: KD=%s, geometry=%s, CKA=%s, Menger=%s\n' \
    "$RUN_NAME" "$KD_RATIO" "$GEOMETRY" "$CKA" "$MENGER_WEIGHT"
: > "$CHECKPOINT_FILE"
CUDA_DEVICES="$CUDA_DEVICES" DATA_DIR="$DATA_DIR" \
    BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" VENV_PATH="$VENV_PATH" \
    CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
    PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" SAVE_PATH="$SAVE_PATH" \
    MAX_LENGTH="$MAX_LENGTH" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    DEV_NUM="$DEV_NUM" SEED="$SEED" \
    KD_LOSS="$KD_LOSS" KD_RATIO="$KD_RATIO" SKEW_ALPHA="$SKEW_ALPHA" \
    GEOMETRY="$GEOMETRY" CKA="$CKA" \
    MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" \
    CKA_WEIGHT="$CKA_WEIGHT" MENGER_WEIGHT="$MENGER_WEIGHT" \
    MENGER_EPS="$MENGER_EPS" DISTILL_TOP_K="$DISTILL_TOP_K" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash "$BASE_PATH/scripts/gemma/train_gemma2_9b_to_2b.sh" "$@"

LORA_PATH="$(<"$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || {
    printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
}

printf '\n[%s 2/2] Evaluate checkpoint: %s\n' "$RUN_NAME" "$LORA_PATH"
CUDA_DEVICES="$CUDA_DEVICES" LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" \
    EVAL_VENV_PATH="$VENV_PATH" EVAL_PYTHON="$VENV_PATH/bin/python" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash "$BASE_PATH/scripts/eval/eval.sh" run
