#!/usr/bin/env bash
set -euo pipefail

# Train the fixed-ratio OFF/SELF/ON variant, then evaluate its final LoRA.
BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_PATH"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    printf 'Training virtualenv not found: %s\n' "$VENV_PATH" >&2
    exit 1
fi
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_DISABLED=true
export TF_CPP_MIN_LOG_LEVEL=3
export CODE_BASE=HF

# Paths follow project_commands.sh and remain overridable from the environment.
CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"
DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

CKPT_NAME="${CKPT_NAME:-qwen2.5-1.5B-Instruct}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen2.5-14B-Instruct}"

# Fixed probabilities. finetune_no_adaptive.py validates that their sum is 1.
export OFF_POLICY_RATIO="${OFF_POLICY_RATIO:-0.90}"
export SELF_DISTILL_RATIO="${SELF_DISTILL_RATIO:-0.20}"
export ON_POLICY_RATIO="${ON_POLICY_RATIO:-0.20}"

# Distributed setup.
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPUS_PER_NODE=${#GPUS[@]}
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# Training hyperparameters, based on scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-5120}"
CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"

KD_LOSS="${KD_LOSS:-sfkl}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
KD_RATIO="${KD_RATIO:-0.5}"
DISTILL_TOP_K="${DISTILL_TOP_K:-512}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
SELF_DISTILL_CONTEXT_DROP_MAX="${SELF_DISTILL_CONTEXT_DROP_MAX:-0.5}"
STEP_SEPARATOR="${STEP_SEPARATOR:-$'\n\n'}"
STEP_POOLING="${STEP_POOLING:-mean}"
MAGNITUDE_NORMALIZATION="${MAGNITUDE_NORMALIZATION:-zscore}"

MAG_WEIGHT="${MAG_WEIGHT:-2.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-2.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
CKA="${CKA:-0}"
DEFAULT_GEOMETRY=1
if [[ "$CKA" == 1 ]]; then
    DEFAULT_GEOMETRY=0
fi
GEOMETRY="${GEOMETRY:-$DEFAULT_GEOMETRY}"

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

RATIO_TAG="off${OFF_POLICY_RATIO}_self${SELF_DISTILL_RATIO}_on${ON_POLICY_RATIO}"
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/${CKPT_NAME}-no-adaptive/${RATIO_TAG}_${KD_LOSS}_k${DISTILL_TOP_K}_geo${GEOMETRY}_cka${CKA}_seed${SEED}}"

for required_path in "$CKPT" "$TEACHER_CKPT" "$DS_CONFIG"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'Required path not found: %s\n' "$required_path" >&2
        exit 1
    fi
done
if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    printf 'Run the preprocessing section in project_commands.sh first, or override DATA_DIR.\n' >&2
    exit 1
fi

OPTS=(
    --base-path "$BASE_PATH"
    --model-path "$CKPT" --model-type qwen --ckpt-name "$CKPT_NAME"
    --teacher-model-path "$TEACHER_CKPT" --teacher-model-type qwen
    --teacher-ckpt-name "$TEACHER_CKPT_NAME"
    --n-gpu "$GPUS_PER_NODE" --n-nodes "$NNODES" --bf16
    --data-dir "$DATA_DIR" --json-data --num-workers "$NUM_WORKERS" --dev-num "$DEV_NUM"
    --lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing
    --lr-decay-style wrmup_cosine --warmup-ratio "$WARMUP_RATIO"
    --weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS"
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH"
    --t-max-length "$T_MAX_LENGTH" --t-max-prompt-length "$T_MAX_PROMPT_LENGTH"
    --type kd --do-sample
    --self-distill-context-drop-ratio "$SELF_DISTILL_CONTEXT_DROP_MAX"
    --self-distill-context-max-tokens "$CONTEXT_MAX_NEW_TOKENS"
    --kd-loss "$KD_LOSS" --kd-ratio "$KD_RATIO" --skew-alpha "$SKEW_ALPHA"
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
if [[ "$GEOMETRY" == 1 ]]; then
    OPTS+=(--geometry)
fi
if [[ "$CKA" == 1 ]]; then
    OPTS+=(--cka)
fi

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT
export FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE"

TRAIN_CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "$BASE_PATH/finetune_no_adaptive.py" "${OPTS[@]}" "$@")
printf '\n[train] fixed-ratio no-adaptive run\n'
printf 'Student model: %s\nTeacher model: %s\nData: %s\nSave: %s\n' \
    "$CKPT" "$TEACHER_CKPT" "$DATA_DIR" "$SAVE_PATH"
printf 'Ratios: OFF=%s SELF=%s ON=%s\n' \
    "$OFF_POLICY_RATIO" "$SELF_DISTILL_RATIO" "$ON_POLICY_RATIO"
printf 'CUDA_VISIBLE_DEVICES=%s\nCommand: ' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

mkdir -p -- "$SAVE_PATH"
"${TRAIN_CMD[@]}"

if [[ ! -s "$CHECKPOINT_FILE" ]]; then
    printf 'Training finished without writing FINAL_CHECKPOINT_FILE: %s\n' "$CHECKPOINT_FILE" >&2
    exit 1
fi
LORA_PATH="$(<"$CHECKPOINT_FILE")"
if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
    printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
fi

printf '\n[eval] final checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES="$CUDA_VISIBLE_DEVICES" \
LORA_PATH="$LORA_PATH" \
MODEL_PATH="$CKPT" \
SAVE_PATH="$(dirname -- "$LORA_PATH")" \
EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-$LORA_R}" \
bash "$BASE_PATH/scripts/eval/eval.sh" run
