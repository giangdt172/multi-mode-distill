#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"
ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
source "$VENV_PATH/bin/activate"
export HF_HOME="${HF_HOME:-$BASE_PATH/.cache/huggingface}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1}}}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NNODES="${NNODES:-1}"
DISTRIBUTED_ARGS=(
    --nproc_per_node "${#GPUS[@]}"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR:-localhost}"
    --master_port "${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
)

CKPT_NAME="gemma-2-2b-it"
TEACHER_CKPT_NAME="gemma-2-9b-it"
CKPT="${CKPT:-google/gemma-2-2b-it}"
TEACHER_CKPT="${TEACHER_CKPT:-google/gemma-2-9b-it}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"
RESOLVED_DATA_DIR="$(PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}" python -c \
    'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
    "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"
DATA_DIR="${DATA_DIR:-$RESOLVED_DATA_DIR}"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-512}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"

KD_LOSS="${KD_LOSS:-sfkl}"
KD_RATIO="${KD_RATIO:-0.5}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
MAG_WEIGHT="${MAG_WEIGHT:-1.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-1.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
CKA="${CKA:-0}"
DEFAULT_GEOMETRY=1
if [[ "$CKA" == 1 ]]; then DEFAULT_GEOMETRY=0; fi
GEOMETRY="${GEOMETRY:-$DEFAULT_GEOMETRY}"
STEP_SEPARATOR="${STEP_SEPARATOR:-$'\n\n'}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
case "$GEOMETRY" in
    0|1) ;;
    *) printf 'GEOMETRY must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$CKA" in
    0|1) ;;
    *) printf 'CKA must be 0 or 1\n' >&2; exit 2 ;;
esac
if [[ "$GEOMETRY" == 1 && "$CKA" == 1 ]]; then
    printf 'GEOMETRY and CKA are mutually exclusive\n' >&2
    exit 2
fi
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/${CKPT_NAME}-distill/adaptive1_${KD_LOSS}_k${DISTILL_TOP_K}_geometry${GEOMETRY}_cka${CKA}_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"

OPTS=(
    --base-path "$BASE_PATH"
    --model-path "$CKPT" --model-type gemma --ckpt-name "$CKPT_NAME"
    --teacher-model-path "$TEACHER_CKPT" --teacher-model-type gemma
    --teacher-ckpt-name "$TEACHER_CKPT_NAME"
    --n-gpu "${#GPUS[@]}" --n-nodes "$NNODES" --bf16
    --data-dir "$DATA_DIR" --json-data --num-workers "$NUM_WORKERS" --dev-num "$DEV_NUM"
    --lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing
    --lr-decay-style wrmup_cosine --warmup-ratio "$WARMUP_RATIO"
    --weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS"
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH"
    --t-max-length "$T_MAX_LENGTH" --t-max-prompt-length "$T_MAX_PROMPT_LENGTH"
    --type kd --kd-loss "$KD_LOSS" --kd-ratio "$KD_RATIO"
    --dual-adaptive-exposure --do-sample
    --rho-self-init "${RHO_SELF_INIT:-0.1}" --rho-on-init "${RHO_ON_INIT:-0.05}"
    --rho-self-max "${RHO_SELF_MAX:-0.25}" --rho-on-max "${RHO_ON_MAX:-0.25}"
    --rho-self-increment "${RHO_SELF_INCREMENT:-0.025}"
    --rho-on-increment "${RHO_ON_INCREMENT:-0.025}"
    --adaptive-threshold "${ADAPTIVE_THRESHOLD:-0.05}"
    --self-distill-eval-seed "${SELF_DISTILL_EVAL_SEED:-1234}"
    --self-distill-context-drop-ratio "${SELF_DISTILL_CONTEXT_DROP_MAX:-0.5}"
    --self-distill-context-max-tokens "$CONTEXT_MAX_NEW_TOKENS"
    --skew-alpha "$SKEW_ALPHA"
    --mag-weight "$MAG_WEIGHT" --gram-weight "$GRAM_WEIGHT" --cka-weight "$CKA_WEIGHT"
    --distill-top-k "$DISTILL_TOP_K" --distill-temperature "$DISTILL_TEMPERATURE"
    --step-separator "$STEP_SEPARATOR" --step-pooling mean
    --magnitude-normalization zscore --eps 1e-6
    --peft lora --peft-lora-r "$LORA_R" --peft-lora-alpha "$LORA_ALPHA"
    --peft-lora-dropout "$LORA_DROPOUT"
    --do-train --do-valid
    --save-interval -1 --eval-interval "$EVAL_INTERVAL" --log-interval 10 --mid-log-num 0
    --save "$SAVE_PATH" --seed "$SEED" --seed-data "$SEED" --seed-lm "$SEED"
    --top-k 0 --top-p 1.0 --temperature 1.0 --repetition-penalty 1.0 --num-beams 1
    --deepspeed --deepspeed_config "$DS_CONFIG"
)
if [[ "$GEOMETRY" == 1 ]]; then
    OPTS+=(--geometry)
fi
if [[ "$CKA" == 1 ]]; then
    OPTS+=(--cka)
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CODE_BASE=HF

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "$BASE_PATH/finetune.py" "${OPTS[@]}" "$@")
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

mkdir -p -- "$SAVE_PATH"
exec "${CMD[@]}"
