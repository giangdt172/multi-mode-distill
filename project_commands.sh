#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

export CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
QWEN_RAW_DATA="${QWEN_RAW_DATA:-$ASSET_ROOT/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"
QWEN_DATA_DIR="${QWEN_DATA_DIR:-${DATA_DIR:-$PROCESSED_DATA_ROOT/$CKPT}}"
QWEN_RESULTS_ROOT="${QWEN_RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2}"

export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}" SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# Reserve the student sequence plus only the extra context budget.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

# Process Qwen data before training.
# printf '\n[process] Qwen data: %s\n' "$QWEN_RAW_DATA"
# python tools/process_data_ultraInteract.py \
#     --base-path "$BASE_PATH" --data-dir "$QWEN_RAW_DATA" \
#     --processed-data-dir "$PROCESSED_DATA_ROOT" \
#     --model-path "$CKPT" --model-type qwen \
#     --data-process-workers "${DATA_PROCESS_WORKERS:-8}" \
#     --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH" \
#     --dev-num "$DEV_NUM" --seed "$SEED"

MAG_WEIGHT="${MAG_WEIGHT:-2.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-2.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-1.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

if [[ ! -s "$QWEN_DATA_DIR/train.jsonl" || ( ! -s "$QWEN_DATA_DIR/valid.jsonl" && ! -s "$QWEN_DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$QWEN_DATA_DIR" >&2
    exit 1
fi

# # CKA comparison: CE + KD + CKA only.
# printf '\n[cka 1/2] Train Qwen: CE + KD + CKA\n'
# : > "$CHECKPOINT_FILE"
# CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" \
#     SAVE_PATH="$QWEN_RESULTS_ROOT/cka" \
#     KD_RATIO="${CE_KD_RATIO:-0.5}" GEOMETRY=0 CKA=1 \
#     FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
#     bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
#         --menger-weight 0 "$@"

# LORA_PATH="$(cat "$CHECKPOINT_FILE")"
# [[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
# printf '\n[cka 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
# CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
#     SAVE_PATH="$(dirname -- "$LORA_PATH")" \
#     EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
#     bash scripts/eval/eval.sh run

# # Menger comparison: CE + KD + step-level Menger curvature only.
# printf '\n[menger 1/2] Train Qwen: CE + KD + Menger (weight=%s, eps=%s)\n' \
#     "$MENGER_WEIGHT" "$MENGER_EPS"
# : > "$CHECKPOINT_FILE"
# CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" \
#     SAVE_PATH="$QWEN_RESULTS_ROOT/menger_weight${MENGER_WEIGHT}" \
#     KD_RATIO="${CE_KD_RATIO:-0.5}" GEOMETRY=0 CKA=0 \
#     FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
#     bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
#         --menger-weight "$MENGER_WEIGHT" --menger-eps "$MENGER_EPS" "$@"

# LORA_PATH="$(cat "$CHECKPOINT_FILE")"
# [[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
# printf '\n[menger 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
# CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
#     SAVE_PATH="$(dirname -- "$LORA_PATH")" \
#     EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
#     bash scripts/eval/eval.sh run

# Qwen setting 1: full objective (CE + KD + stronger geometry).
# 1. Train synchronously using the existing processed data.
printf '\n[full 1/2] Train Qwen: CE + KD + geometry (mag=%s, gram=%s)\n' "$MAG_WEIGHT" "$GRAM_WEIGHT"
: > "$CHECKPOINT_FILE"
CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" \
    SAVE_PATH="$QWEN_RESULTS_ROOT/full_mag${MAG_WEIGHT}_gram${GRAM_WEIGHT}" \
    KD_RATIO="${CE_KD_RATIO:-0.5}" GEOMETRY=1 CKA=0 \
    MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

# 2. Evaluate this run's final checkpoint only after training succeeds.
LORA_PATH="$(cat "$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
printf '\n[full 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run

# Qwen setting 2: remove geometry (CE + KD only).
# 1. Train synchronously using the existing processed data.
# printf '\n[no-geo 1/2] Train Qwen: CE + KD, geometry disabled\n'
# : > "$CHECKPOINT_FILE"
# CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" \
#     SAVE_PATH="$QWEN_RESULTS_ROOT/no_geo" \
#     KD_RATIO="${CE_KD_RATIO:-0.5}" GEOMETRY=0 CKA=0 \
#     FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
#     bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
#         --menger-weight 0 "$@"

# # 2. Evaluate this run's final checkpoint only after training succeeds.
# LORA_PATH="$(cat "$CHECKPOINT_FILE")"
# [[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
# printf '\n[no-geo 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
# CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
#     SAVE_PATH="$(dirname -- "$LORA_PATH")" \
#     EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
#     bash scripts/eval/eval.sh run

# # Qwen setting 3: remove CE (KD + stronger geometry only).
# # 1. Train synchronously using the existing processed data.
# printf '\n[no-CE 1/2] Train Qwen: KD + geometry, CE disabled (mag=%s, gram=%s)\n' "$MAG_WEIGHT" "$GRAM_WEIGHT"
# : > "$CHECKPOINT_FILE"
# CUDA_DEVICES=4,5,6,7 DATA_DIR="$QWEN_DATA_DIR" \
#     SAVE_PATH="$QWEN_RESULTS_ROOT/no_ce_mag${MAG_WEIGHT}_gram${GRAM_WEIGHT}" \
#     KD_RATIO=1.0 GEOMETRY=1 CKA=0 \
#     MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" \
#     FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
#     bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
#         --disable-lm-loss --menger-weight 0 "$@"

# # 2. Evaluate this run's final checkpoint only after training succeeds.
# LORA_PATH="$(cat "$CHECKPOINT_FILE")"
# [[ -f "$LORA_PATH/adapter_config.json" ]] || { printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2; exit 1; }
# printf '\n[no-CE 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
# CUDA_DEVICES=4,5,6,7 LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
#     SAVE_PATH="$(dirname -- "$LORA_PATH")" \
#     EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
#     bash scripts/eval/eval.sh run

# Ablation fixed OFF/SELF/ON exposure, without adaptive ratio updates.
# run_no_adaptive.sh performs both training and final-checkpoint evaluation.

# NO_ADAPTIVE_OFF_RATIO="${NO_ADAPTIVE_OFF_RATIO:-${OFF_POLICY_RATIO:-0.90}}"
# NO_ADAPTIVE_SELF_RATIO="${NO_ADAPTIVE_SELF_RATIO:-${SELF_DISTILL_RATIO:-0.05}}"
# NO_ADAPTIVE_ON_RATIO="${NO_ADAPTIVE_ON_RATIO:-${ON_POLICY_RATIO:-0.05}}"
# NO_ADAPTIVE_SAVE_PATH="${NO_ADAPTIVE_SAVE_PATH:-$QWEN_RESULTS_ROOT/no_adaptive_off${NO_ADAPTIVE_OFF_RATIO}_self${NO_ADAPTIVE_SELF_RATIO}_on${NO_ADAPTIVE_ON_RATIO}}"

# printf '\n[no-adaptive] Train + evaluate fixed-ratio Qwen: OFF=%s, SELF=%s, ON=%s\n' \
#     "$NO_ADAPTIVE_OFF_RATIO" "$NO_ADAPTIVE_SELF_RATIO" "$NO_ADAPTIVE_ON_RATIO"
# CUDA_DEVICES=4,5,6,7 \
#     CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" DATA_DIR="$QWEN_DATA_DIR" \
#     SAVE_PATH="$NO_ADAPTIVE_SAVE_PATH" \
#     OFF_POLICY_RATIO="$NO_ADAPTIVE_OFF_RATIO" \
#     SELF_DISTILL_RATIO="$NO_ADAPTIVE_SELF_RATIO" \
#     ON_POLICY_RATIO="$NO_ADAPTIVE_ON_RATIO" \
#     KD_RATIO="${CE_KD_RATIO:-0.5}" GEOMETRY=0 CKA=0 \
#     MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" \
#     bash "$BASE_PATH/run_no_adaptive.sh" "$@"
