#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"
export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$BASE_PATH/multi-mode-distill.txt}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    command -v uv >/dev/null || {
        printf 'uv is required to create %s\n' "$VENV_PATH" >&2
        exit 1
    }
    [[ -f "$REQUIREMENTS_FILE" ]] || {
        printf 'Requirements file not found: %s\n' "$REQUIREMENTS_FILE" >&2
        exit 1
    }
    uv venv --python 3.11 "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    uv pip install -r "$REQUIREMENTS_FILE"
else
    source "$VENV_PATH/bin/activate"
fi
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$BASE_PATH/.cache/huggingface}"

export CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
RAW_DATA="${RAW_DATA:-$ASSET_ROOT/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"

RESOLVED_DATA_DIR="$(python -c \
    'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
    "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"
DATA_DIR="${DATA_DIR:-$RESOLVED_DATA_DIR}"
export DATA_DIR
export MAX_LENGTH="${MAX_LENGTH:-1024}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}" SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# Reserve the student sequence plus only the extra context budget.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

mkdir -p -- "$(dirname -- "$RAW_DATA")" "$PROCESSED_DATA_ROOT" "$HF_HOME"
if [[ ! -s "$RAW_DATA" ]]; then
    hf download VoCuc/UltraInteract-Infer \
        Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
        --repo-type dataset \
        --local-dir "$ASSET_ROOT/data/raw"
fi

mkdir -p -- "$EVAL_DATA_DIR/code_eval"
hf download openai/gsm8k --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm8k"
hf download qintongli/GSM-Plus --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm_plus"
hf download EleutherAI/hendrycks_math --repo-type dataset --local-dir "$EVAL_DATA_DIR/hendrycks_math"
hf download google-research-datasets/mbpp --repo-type dataset --local-dir "$EVAL_DATA_DIR/mbpp"
hf download allenai/sciq --repo-type dataset --local-dir "$EVAL_DATA_DIR/sciq"
hf download cais/mmlu --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu"
hf download TIGER-Lab/MMLU-Pro --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu_pro"
hf download SaylorTwift/bbh --repo-type dataset --local-dir "$EVAL_DATA_DIR/bbh"
curl --fail --location \
    https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/code_eval.py \
    --output "$EVAL_DATA_DIR/code_eval/code_eval.py"
curl --fail --location \
    https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/execute.py \
    --output "$EVAL_DATA_DIR/code_eval/execute.py"

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}" \
    python "$BASE_PATH/tools/process_data_ultraInteract.py" \
        --base-path "$BASE_PATH" \
        --data-dir "$RAW_DATA" \
        --processed-data-dir "$PROCESSED_DATA_ROOT" \
        --model-path "$CKPT" \
        --model-type qwen \
        --data-process-workers "${DATA_PROCESS_WORKERS:-32}" \
        --max-length "$MAX_LENGTH" \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --dev-num "$DEV_NUM" \
        --seed "$SEED"
fi

# Pairwise adaptive probabilities: ON changes in on_self; SELF changes in off_self.
# The other enabled mode always receives the complementary probability.
export RHO_SELF_INIT=0.10
export RHO_ON_INIT=0.05
export RHO_SELF_MAX=0.25
export RHO_ON_MAX=0.25
export RHO_SELF_INCREMENT=0.025
export RHO_ON_INCREMENT=0.025

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
fi

MODE_CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$MODE_CHECKPOINT_FILE"' EXIT
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/adaptive_mode_ablation}"
read -r -a ADAPTIVE_ABLATIONS <<< "${ADAPTIVE_ABLATIONS:-on_self off_self}"

for ADAPTIVE_MODE_SET in "${ADAPTIVE_ABLATIONS[@]}"; do
    case "$ADAPTIVE_MODE_SET" in
        on_self|off_self) ;;
        *) printf 'Unsupported adaptive ablation: %s\n' "$ADAPTIVE_MODE_SET" >&2; exit 2 ;;
    esac

    printf '\n[%s 1/2] Train adaptive Qwen with modes: %s\n' \
        "$ADAPTIVE_MODE_SET" "$ADAPTIVE_MODE_SET"
    : > "$MODE_CHECKPOINT_FILE"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
        DATA_DIR="$DATA_DIR" ADAPTIVE_MODE_SET="$ADAPTIVE_MODE_SET" \
        SAVE_PATH="$RESULTS_ROOT/$ADAPTIVE_MODE_SET" \
        FINAL_CHECKPOINT_FILE="$MODE_CHECKPOINT_FILE" \
        bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

    MODE_LORA_PATH="$(cat "$MODE_CHECKPOINT_FILE")"
    [[ -f "$MODE_LORA_PATH/adapter_config.json" ]] || {
        printf 'Final LoRA checkpoint missing: %s\n' "$MODE_LORA_PATH" >&2
        exit 1
    }
    printf '\n[%s 2/2] Evaluate checkpoint: %s\n' \
        "$ADAPTIVE_MODE_SET" "$MODE_LORA_PATH"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
        LORA_PATH="$MODE_LORA_PATH" MODEL_PATH="$CKPT" \
        EVAL_VENV_PATH="$VENV_PATH" EVAL_PYTHON="$VENV_PATH/bin/python" \
        EVAL_DATA_DIR="$EVAL_DATA_DIR" \
        SAVE_PATH="$(dirname -- "$MODE_LORA_PATH")" \
        EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
        bash scripts/eval/eval.sh run
done
