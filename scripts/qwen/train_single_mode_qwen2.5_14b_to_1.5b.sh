#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
    off_policy|on_policy|self_distill|opsd) shift ;;
    *) printf 'Usage: %s {off_policy|on_policy|self_distill|opsd} [finetune arguments...]\n' "$0" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1}}}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPUS_PER_NODE=${#GPUS[@]}
NNODES="${NNODES:-1}"
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR:-localhost}"
    --master_port "${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
)

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"
ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
export HF_HOME="${HF_HOME:-$BASE_PATH/.cache/huggingface}"
CKPT_NAME="qwen2.5-1.5B-Instruct"
TEACHER_CKPT_NAME="qwen2.5-14B-Instruct"
CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
DATA_DIR="${DATA_DIR:-$ASSET_ROOT/data/processed/ultraInteract-v2/Qwen/Qwen2.5-1.5B-Instruct}"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"

KD_LOSS="${KD_LOSS:-sfkl}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
if [[ "$MODE" == off_policy || "$MODE" == self_distill ]]; then
    KD_RATIO="${KD_RATIO:-0.5}"
else
    KD_RATIO="${KD_RATIO:-1.0}"
fi
MAG_WEIGHT="${MAG_WEIGHT:-1.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-1.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
CKA="${CKA:-0}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
SELF_DISTILL_CONTEXT_DROP_MAX="${SELF_DISTILL_CONTEXT_DROP_MAX:-0.5}"
SELF_DISTILL_CONTEXT_FIXED_DROP_RATIO="${SELF_DISTILL_CONTEXT_FIXED_DROP_RATIO:-}"
STEP_SEPARATOR="${STEP_SEPARATOR:-$'\n\n'}"
STEP_POOLING="${STEP_POOLING:-mean}"
MAGNITUDE_NORMALIZATION="${MAGNITUDE_NORMALIZATION:-zscore}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DEFAULT_GEOMETRY=0
GEOMETRY="${GEOMETRY:-$DEFAULT_GEOMETRY}"
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
if [[ "$MODE" == opsd && ("$GEOMETRY" == 1 || "$CKA" == 1) ]]; then
    printf 'Geometry and CKA are unavailable for opsd\n' >&2
    exit 2
fi

SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/${CKPT_NAME}-${MODE}/${KD_LOSS}_k${DISTILL_TOP_K}_geometry${GEOMETRY}_cka${CKA}_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"

OPTS=(
    --base-path "$BASE_PATH"
    --model-path "$CKPT" --model-type qwen --ckpt-name "$CKPT_NAME"
    --n-gpu "$GPUS_PER_NODE" --n-nodes "$NNODES" --bf16
    --data-dir "$DATA_DIR" --json-data --num-workers "$NUM_WORKERS" --dev-num "$DEV_NUM"
    --lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing
    --lr-decay-style wrmup_cosine --warmup-ratio "$WARMUP_RATIO"
    --weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS"
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH"
    --t-max-length "$T_MAX_LENGTH" --t-max-prompt-length "$T_MAX_PROMPT_LENGTH"
    --type kd --distill-mode "$MODE" --kd-loss "$KD_LOSS" --kd-ratio "$KD_RATIO"
    --skew-alpha "$SKEW_ALPHA"
    --mag-weight "$MAG_WEIGHT" --gram-weight "$GRAM_WEIGHT" --cka-weight "$CKA_WEIGHT"
    --distill-top-k "$DISTILL_TOP_K" --distill-temperature "$DISTILL_TEMPERATURE"
    --step-separator "$STEP_SEPARATOR" --step-pooling "$STEP_POOLING"
    --magnitude-normalization "$MAGNITUDE_NORMALIZATION" --eps 1e-6
    --peft lora --peft-lora-r "$LORA_R" --peft-lora-alpha "$LORA_ALPHA"
    --peft-lora-dropout "$LORA_DROPOUT"
    --do-train --do-valid
    --save-interval -1 --eval-interval "$EVAL_INTERVAL" --log-interval 10 --mid-log-num 0
    --save "$SAVE_PATH" --seed "$SEED" --seed-data "$SEED" --seed-lm "$SEED"
    --top-k 0 --top-p 1.0 --temperature 1.0 --repetition-penalty 1.0 --num-beams 1
    --deepspeed --deepspeed_config "$DS_CONFIG"
)

if [[ "$MODE" != self_distill && "$MODE" != opsd ]]; then
    OPTS+=(--teacher-model-path "$TEACHER_CKPT" --teacher-model-type qwen
           --teacher-ckpt-name "$TEACHER_CKPT_NAME")
fi
if [[ "$MODE" == on_policy || "$MODE" == opsd ]]; then
    OPTS+=(--do-sample)
fi
if [[ "$GEOMETRY" == 1 ]]; then
    OPTS+=(--geometry)
fi
if [[ "$CKA" == 1 ]]; then
    OPTS+=(--cka)
fi
if [[ "$MODE" == self_distill ]]; then
    OPTS+=(--self-distill-context-drop-ratio "$SELF_DISTILL_CONTEXT_DROP_MAX"
           --self-distill-context-max-tokens "$CONTEXT_MAX_NEW_TOKENS")
    if [[ -n "$SELF_DISTILL_CONTEXT_FIXED_DROP_RATIO" ]]; then
        OPTS+=(--self-distill-context-fixed-drop-ratio
               "$SELF_DISTILL_CONTEXT_FIXED_DROP_RATIO")
    fi
fi
if [[ "$MODE" == opsd ]]; then
    OPTS+=(--disable-lm-loss --opsd-token-clip "${OPSD_TOKEN_CLIP:-0.05}")
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

if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
mkdir -p -- "$SAVE_PATH"
exec "${CMD[@]}"
