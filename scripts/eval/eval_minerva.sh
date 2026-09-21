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


TASK_MINERVA="minerva_math"

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

case "${1:-run}" in
    check)
        "$PYTHON_BIN" scripts/eval/local_lm_eval.py check --tasks "$TASK_MINERVA"
        exit 0
        ;;
    run|run_eval)
        # Usage: eval_minerva.sh run_eval /absolute/path/to/lora
        LORA_PATH="${2:-$LORA_PATH}"
        ;;
    *)
        printf 'Usage: %s [run_eval [LORA_PATH] | check]\n' "$0" >&2
        exit 2
        ;;
esac

if [[ -z "$LORA_PATH" ]]; then
    FINAL_STEP="$($PYTHON_BIN -c 'import json, sys; print(json.load(open(sys.argv[1]))["total_iters"])' "$SAVE_PATH/args.json")"
    LORA_PATH="$SAVE_PATH/$FINAL_STEP"
fi

MODEL_ARGS="pretrained=$MODEL_PATH,dtype=$DTYPE,max_length=$MAX_MODEL_LENGTH"
MODEL_ARGS+=",tensor_parallel_size=1,data_parallel_size=$DATA_PARALLEL_SIZE"
MODEL_ARGS+=",enable_prefix_caching=True,enable_chunked_prefill=True"

BASE_ARGS=(
    --model vllm
    --batch_size auto
    --log_samples
    --apply_chat_template
    --fewshot_as_multiturn
    --output_path "$OUT/general"
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,temperature=0.0"
)

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "data_parallel_size=$DATA_PARALLEL_SIZE"

run_eval() {
    local tasks="$1"
    local lora_path="$2"
    shift 2
    echo "LoRA=$lora_path"
    "$PYTHON_BIN" scripts/eval/local_lm_eval.py run --tasks "$tasks" \
        --model_args "$MODEL_ARGS,lora_local_path=$lora_path,max_lora_rank=$MAX_LORA_RANK" "$@"
}


run_eval "$TASK_MINERVA" "$LORA_PATH_1" "${BASE_ARGS[@]}"
run_eval "$TASK_MINERVA" "$LORA_PATH_2" "${BASE_ARGS[@]}"
