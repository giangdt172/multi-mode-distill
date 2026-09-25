#!/usr/bin/env bash
set -euo pipefail

# Gemma 2 evaluation entrypoint.
# Unlike lm-eval's default fallback, this preserves system instructions by
# prepending them to the first user turn before applying Gemma's chat template.

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
unset PYTHONPATH

ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
EVAL_VENV_PATH="${EVAL_VENV_PATH:-$BASE_PATH/.venv}"
PYTHON_BIN="${EVAL_PYTHON:-$EVAL_VENV_PATH/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5}}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-google/gemma-2-2b-it}}"
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/gemma-2-2b-it-distill}"
LORA_PATH="${LORA_PATH:-}"
OUT="${EVAL_OUTPUT_DIR:-$SAVE_PATH/evaluation_gemma}"

DTYPE="${EVAL_DTYPE:-bfloat16}"
MAX_MODEL_LENGTH="${EVAL_MAX_LENGTH:-8192}"
MAX_GEN_TOKS="${EVAL_MAX_NEW_TOKENS:-5120}"
MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-16}"
USE_LORA="${EVAL_USE_LORA:-1}"

DEFAULT_TASKS="gsm8k,minerva_math,mmlu_stem,gsm_plus,mbpp"
EVAL_TASKS="${EVAL_TASKS:-$DEFAULT_TASKS}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
DATA_PARALLEL_SIZE=${#GPU_LIST[@]}

export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
export HF_HOME="${EVAL_HF_HOME:-$BASE_PATH/.cache/huggingface}"
export HF_DATASETS_CACHE="${EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$BASE_PATH/.cache/eval/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export EVAL_GEMMA_MERGE_SYSTEM=1
export MBPP_CLEAN_SENTENCEPIECE="${MBPP_CLEAN_SENTENCEPIECE:-1}"

mkdir -p "$OUT" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

case "${1:-run}" in
    check)
        "$PYTHON_BIN" scripts/eval/local_lm_eval.py check --tasks "$EVAL_TASKS"
        exit 0
        ;;
    run)
        ;;
    *)
        printf 'Usage: %s [run | check]\n' "$0" >&2
        exit 2
        ;;
esac

case "$USE_LORA" in
    0|1) ;;
    *) printf 'EVAL_USE_LORA must be 0 or 1\n' >&2; exit 2 ;;
esac

MODEL_ARGS="pretrained=$MODEL_PATH,dtype=$DTYPE,max_length=$MAX_MODEL_LENGTH"
MODEL_ARGS+=",tensor_parallel_size=1,data_parallel_size=$DATA_PARALLEL_SIZE"
MODEL_ARGS+=",enable_prefix_caching=True,enable_chunked_prefill=True"

if [[ "$USE_LORA" == 1 ]]; then
    if [[ -z "$LORA_PATH" ]]; then
        ARGS_FILE="$SAVE_PATH/args.json"
        [[ -f "$ARGS_FILE" ]] || {
            printf 'Missing %s; set LORA_PATH or use EVAL_USE_LORA=0 for the base model.\n' \
                "$ARGS_FILE" >&2
            exit 1
        }
        FINAL_STEP="$($PYTHON_BIN -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_iters"])' "$ARGS_FILE")"
        LORA_PATH="$SAVE_PATH/$FINAL_STEP"
    fi
    [[ -f "$LORA_PATH/adapter_config.json" ]] || {
        printf 'Missing LoRA adapter: %s/adapter_config.json\n' "$LORA_PATH" >&2
        exit 1
    }
    MODEL_ARGS+=",lora_local_path=$LORA_PATH,max_lora_rank=$MAX_LORA_RANK"
fi

COMMON_ARGS=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

CHAT_ARGS=(
    "${COMMON_ARGS[@]}"
    --apply_chat_template
    --fewshot_as_multiturn
    --output_path "$OUT/general"
)

CODE_ARGS=(
    "${COMMON_ARGS[@]}"
    --apply_chat_template
    --fewshot_as_multiturn
    --confirm_run_unsafe_code
    --output_path "$OUT/code"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

run_task() {
    local task="$1"
    case "$task" in
        mbpp)
            "$PYTHON_BIN" scripts/eval/local_lm_eval.py run \
                --tasks "$task" "${CODE_ARGS[@]}"
            ;;
        mmlu_stem)
            "$PYTHON_BIN" scripts/eval/local_lm_eval.py run \
                --tasks "$task" --num_fewshot 5 "${CHAT_ARGS[@]}"
            ;;
        gsm8k|minerva_math|sciq|bbh_cot_fewshot|gsm_plus|mmlu_pro_math)
            "$PYTHON_BIN" scripts/eval/local_lm_eval.py run \
                --tasks "$task" "${CHAT_ARGS[@]}"
            ;;
        *)
            printf 'Unsupported Gemma eval task: %s\n' "$task" >&2
            exit 2
            ;;
    esac
}

printf 'Gemma model: %s\n' "$MODEL_PATH"
printf 'CUDA_VISIBLE_DEVICES=%s (data_parallel_size=%s)\n' \
    "$CUDA_VISIBLE_DEVICES" "$DATA_PARALLEL_SIZE"
if [[ "$USE_LORA" == 1 ]]; then
    printf 'LoRA: %s\n' "$LORA_PATH"
else
    printf 'LoRA: disabled (base-model evaluation)\n'
fi
printf 'Tasks: %s\n' "$EVAL_TASKS"

IFS=',' read -r -a TASK_LIST <<< "$EVAL_TASKS"
for task in "${TASK_LIST[@]}"; do
    [[ -n "$task" ]] || continue
    run_task "$task"
done
