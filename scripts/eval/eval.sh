#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$BASE_PATH"
unset PYTHONPATH
ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"

EVAL_VENV_PATH="${EVAL_VENV_PATH:-/mnt/local/uvenvs/multi-mode-distill-eval}"
PYTHON_BIN="${EVAL_PYTHON:-$EVAL_VENV_PATH/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-4,5}}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}}"
SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-multi-mode}"
LORA_PATH="${LORA_PATH:-}"
OUT="${EVAL_OUTPUT_DIR:-$SAVE_PATH/evaluation}"

DTYPE="${EVAL_DTYPE:-bfloat16}"
MAX_MODEL_LENGTH="${EVAL_MAX_LENGTH:-8192}"
MAX_GEN_TOKS="${EVAL_MAX_NEW_TOKENS:-5120}"
MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-16}"

TASK_SCIQ="sciq"
TASK_BBH="bbh_cot_fewshot"
TASK_MBPP="mbpp"
TASK_MBPP_INSTRUCT="mbpp_instruct"
TASK_GSM8K="gsm8k"
TASK_GSM_PLUS="gsm_plus"
TASK_MINERVA="minerva_math"
TASK_MMLU_PRO_MATH="mmlu_pro_math"
TASK_MMLU="mmlu_stem"
TASK_Hendrycks_Math="hendrycks_math"
ALL_TASKS="$TASK_GSM8K,$TASK_MINERVA,$TASK_SCIQ,$TASK_BBH,$TASK_MMLU,$TASK_GSM_PLUS,$TASK_MMLU_PRO_MATH,$TASK_MBPP,$TASK_Hendrycks_Math"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
DATA_PARALLEL_SIZE=${#GPU_LIST[@]}

export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
export HF_HOME="${EVAL_HF_HOME:-$BASE_PATH/.cache/eval/huggingface}"
export HF_DATASETS_CACHE="${EVAL_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_MODULES_CACHE="${EVAL_MODULES_CACHE:-$BASE_PATH/.cache/eval/modules}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE"

if [[ "${1:-run}" == "check" ]]; then
    "$PYTHON_BIN" scripts/eval/local_lm_eval.py check --tasks "$ALL_TASKS"
    exit 0
fi

if [[ -z "$LORA_PATH" ]]; then
    FINAL_STEP="$($PYTHON_BIN -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_iters"])' "$SAVE_PATH/args.json")"
    LORA_PATH="$SAVE_PATH/$FINAL_STEP"
fi

MODEL_ARGS="pretrained=$MODEL_PATH,dtype=$DTYPE,max_length=$MAX_MODEL_LENGTH"
MODEL_ARGS+=",tensor_parallel_size=1,data_parallel_size=$DATA_PARALLEL_SIZE"
MODEL_ARGS+=",enable_prefix_caching=True,enable_chunked_prefill=True"
MODEL_ARGS+=",lora_local_path=$LORA_PATH,max_lora_rank=$MAX_LORA_RANK"

BASE_ARGS=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --apply_chat_template
    --fewshot_as_multiturn
    --output_path "$OUT/general"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

BASE_ARGS_no_chat=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    # --fewshot_as_multiturn
    --log_samples
    --output_path "$OUT/general"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

# MBPP: no chat template and no fewshot_as_multiturn.
BASE_ARGS_CODE=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --confirm_run_unsafe_code
    --output_path "$OUT/code"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

BASE_ARGS_CODE_1=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --confirm_run_unsafe_code
    --apply_chat_template
    --output_path "$OUT/code"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

# Math: no fewshot_as_multiturn.
BASE_ARGS_MATH=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --output_path "$OUT/math"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

BASE_ARGS_MATH_1=(
    --model vllm
    --model_args "$MODEL_ARGS"
    --batch_size auto
    --log_samples
    --fewshot_as_multiturn
    --output_path "$OUT/math"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "data_parallel_size=$DATA_PARALLEL_SIZE"
echo "LoRA=$LORA_PATH"

run_eval() {
    local tasks="$1"
    shift
    "$PYTHON_BIN" scripts/eval/local_lm_eval.py run --tasks "$tasks" "$@"
}


run_eval "$TASK_GSM8K" "${BASE_ARGS[@]}"
run_eval "$TASK_MINERVA" "${BASE_ARGS[@]}"
run_eval "$TASK_SCIQ" "${BASE_ARGS[@]}"
run_eval "$TASK_BBH" "${BASE_ARGS[@]}"
run_eval "$TASK_MMLU" --num_fewshot 5 "${BASE_ARGS[@]}"
run_eval "$TASK_GSM_PLUS" "${BASE_ARGS[@]}"
run_eval "$TASK_MMLU_PRO_MATH" "${BASE_ARGS[@]}"

# Code
run_eval "$TASK_MBPP" "${BASE_ARGS_CODE[@]}"
# printf 'Run setting 2 for code\n'
# run_eval "$TASK_MBPP" "${BASE_ARGS_CODE_1[@]}"
