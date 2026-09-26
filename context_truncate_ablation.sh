#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_PATH
cd "$BASE_PATH"

export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
export CKPT="${CKPT:-Qwen/Qwen2.5-1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen2.5-14B-Instruct}"
export HF_HOME="${HF_HOME:-$BASE_PATH/.cache/huggingface}"
RAW_DATA="${RAW_DATA:-$ASSET_ROOT/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"
export DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/$CKPT}"
export CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}"
export SEED="${SEED:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

# Keep adaptive OFF + SELF + ON unchanged. Only the fixed fraction of trailing
# SELF context steps is ablated.
CONTEXT_TRUNCATE_RATIOS=(0.00 0.25 0.50)
ADAPTIVE_MODE_SET=all
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2/context_truncate_ablation}"

CKA="${CKA:-0}"
DEFAULT_GEOMETRY=1
if [[ "$CKA" == 1 ]]; then DEFAULT_GEOMETRY=0; fi
GEOMETRY="${GEOMETRY:-$DEFAULT_GEOMETRY}"
MAG_WEIGHT="${MAG_WEIGHT:-1.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-1.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-0.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"

VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
[[ -f "$VENV_PATH/bin/activate" ]] || {
    printf 'Virtual environment is required at: %s\n' "$VENV_PATH" >&2
    exit 1
}
source "$VENV_PATH/bin/activate"

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    mkdir -p -- "$(dirname -- "$RAW_DATA")" "$PROCESSED_DATA_ROOT" "$HF_HOME"
    if [[ ! -s "$RAW_DATA" ]]; then
        hf download VoCuc/UltraInteract-Infer \
            Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl \
            --repo-type dataset \
            --local-dir "$ASSET_ROOT/data/raw"
    fi
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

[[ -s "$DATA_DIR/train.jsonl" && ( -s "$DATA_DIR/valid.jsonl" || -s "$DATA_DIR/dev.jsonl" ) ]] || {
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
}

mkdir -p -- "$EVAL_DATA_DIR/code_eval"
hf download openai/gsm8k --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm8k"
hf download qintongli/GSM-Plus --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm_plus"
hf download EleutherAI/hendrycks_math --repo-type dataset --local-dir "$EVAL_DATA_DIR/hendrycks_math"
hf download google-research-datasets/mbpp --repo-type dataset --local-dir "$EVAL_DATA_DIR/mbpp"
hf download allenai/sciq --repo-type dataset --local-dir "$EVAL_DATA_DIR/sciq"
hf download cais/mmlu --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu"
hf download TIGER-Lab/MMLU-Pro --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu_pro"
hf download SaylorTwift/bbh --repo-type dataset --local-dir "$EVAL_DATA_DIR/bbh"
[[ -s "$EVAL_DATA_DIR/code_eval/code_eval.py" ]] || curl --fail --location \
    https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/code_eval.py \
    --output "$EVAL_DATA_DIR/code_eval/code_eval.py"
[[ -s "$EVAL_DATA_DIR/code_eval/execute.py" ]] || curl --fail --location \
    https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/execute.py \
    --output "$EVAL_DATA_DIR/code_eval/execute.py"

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

for RATIO in "${CONTEXT_TRUNCATE_RATIOS[@]}"; do
    RATIO_TAG="${RATIO//./p}"
    RUN_SAVE_PATH="$RESULTS_ROOT/drop_${RATIO_TAG}"
    printf '\n[drop=%s 1/2] Train with a fixed SELF context truncation ratio\n' "$RATIO"
    : > "$CHECKPOINT_FILE"
    ADAPTIVE_MODE_SET="$ADAPTIVE_MODE_SET" \
        SAVE_PATH="$RUN_SAVE_PATH" FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
        GEOMETRY="$GEOMETRY" CKA="$CKA" \
        MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" CKA_WEIGHT="$CKA_WEIGHT" \
        MENGER_WEIGHT="$MENGER_WEIGHT" MENGER_EPS="$MENGER_EPS" \
        bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh \
        "$@" --self-distill-context-fixed-drop-ratio "$RATIO"

    LORA_PATH="$(cat "$CHECKPOINT_FILE")"
    [[ -f "$LORA_PATH/adapter_config.json" ]] || {
        printf 'Final LoRA checkpoint missing for ratio %s: %s\n' \
            "$RATIO" "$LORA_PATH" >&2
        exit 1
    }
    printf '\n[drop=%s 2/2] Evaluate checkpoint: %s\n' "$RATIO" "$LORA_PATH"
    LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
        EVAL_VENV_PATH="$VENV_PATH" EVAL_PYTHON="$VENV_PATH/bin/python" \
        EVAL_DATA_DIR="$EVAL_DATA_DIR" \
        SAVE_PATH="$(dirname -- "$LORA_PATH")" \
        EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
        bash scripts/eval/eval.sh run
done
