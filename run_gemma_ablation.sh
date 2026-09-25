#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE_PATH
cd "$BASE_PATH"

export ASSET_ROOT="${ASSET_ROOT:-$BASE_PATH}"
VENV_PATH="${VENV_PATH:-$BASE_PATH/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$BASE_PATH/multi-mode-distill.txt}"

source "$VENV_PATH/bin/activate"
# if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
#     command -v uv >/dev/null || {
#         printf 'uv is required to create %s\n' "$VENV_PATH" >&2
#         exit 1
#     }
#     [[ -f "$REQUIREMENTS_FILE" ]] || {
#         printf 'Requirements file not found: %s\n' "$REQUIREMENTS_FILE" >&2
#         exit 1
#     }
#     uv venv --python 3.11 "$VENV_PATH"
#     source "$VENV_PATH/bin/activate"
#     uv pip install -r "$REQUIREMENTS_FILE"
# else
#     source "$VENV_PATH/bin/activate"
# fi

export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$BASE_PATH/.cache/huggingface}"

export CKPT="${CKPT:-google/gemma-2-2b-it}"
export TEACHER_CKPT="${TEACHER_CKPT:-google/gemma-2-9b-it}"
RAW_DATA="${RAW_DATA:-$ASSET_ROOT/data/raw/google/gemma-2-9b-it/generated_train.jsonl}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/data/processed/ultraInteract-v2}"

RESOLVED_DATA_DIR="$(python -c \
    'import sys; from tools.process_data_ultraInteract import resolve_processed_data_dir; print(resolve_processed_data_dir(*sys.argv[1:]))' \
    "$PROCESSED_DATA_ROOT" "$CKPT" "$BASE_PATH")"
DATA_DIR="${DATA_DIR:-$RESOLVED_DATA_DIR}"
export DATA_DIR
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}"
export SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

KD_LOSS="${KD_LOSS:-fkl}"
KD_RATIO="${KD_RATIO:-0.5}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
DISTILL_TOP_K="${DISTILL_TOP_K:-10240}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
FINETUNE_ENTRYPOINT="${FINETUNE_ENTRYPOINT:-finetune.py}"
RUN_EVAL="${RUN_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

GEOMETRY="${GEOMETRY:-0}"
CKA="${CKA:-0}"
MAG_WEIGHT="${MAG_WEIGHT:-0.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-0.0}"
CKA_WEIGHT="${CKA_WEIGHT:-0.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-0.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"
SELF_DISTILL_CONTEXT_DROP_MAX="${SELF_DISTILL_CONTEXT_DROP_MAX:-0.5}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-0.05}"
SELF_DISTILL_EVAL_SEED="${SELF_DISTILL_EVAL_SEED:-1234}"

case "$RUN_EVAL" in
    0|1) ;;
    *) printf 'RUN_EVAL must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$DRY_RUN" in
    0|1) ;;
    *) printf 'DRY_RUN must be 0 or 1\n' >&2; exit 2 ;;
esac
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
if [[ "$FINETUNE_ENTRYPOINT" == /* ]]; then
    FINETUNE_PATH="$FINETUNE_ENTRYPOINT"
else
    FINETUNE_PATH="$BASE_PATH/$FINETUNE_ENTRYPOINT"
fi
if [[ ! -f "$FINETUNE_PATH" ]]; then
    printf 'Finetune entrypoint not found: %s\n' "$FINETUNE_PATH" >&2
    exit 1
fi

read -r -a ADAPTIVE_ABLATIONS <<< "${ADAPTIVE_ABLATIONS:-on_self off_self self_only}"
if (( ${#ADAPTIVE_ABLATIONS[@]} == 0 )); then
    printf 'ADAPTIVE_ABLATIONS must contain at least one mode\n' >&2
    exit 2
fi
for ADAPTIVE_MODE_SET in "${ADAPTIVE_ABLATIONS[@]}"; do
    case "$ADAPTIVE_MODE_SET" in
        on_self|off_self|self_only) ;;
        *) printf 'Unsupported Gemma ablation: %s\n' "$ADAPTIVE_MODE_SET" >&2; exit 2 ;;
    esac
done

# if [[ "$DRY_RUN" == 0 ]]; then
#     mkdir -p -- "$(dirname -- "$RAW_DATA")" "$PROCESSED_DATA_ROOT" "$HF_HOME"
#     if [[ ! -s "$RAW_DATA" ]]; then
#         hf download VoCuc/UltraInteract-Infer \
#             google/gemma-2-9b-it/generated_train.jsonl \
#             --repo-type dataset \
#             --local-dir "$ASSET_ROOT/data/raw"
#     fi
# fi

# if [[ "$DRY_RUN" == 0 && "$RUN_EVAL" == 1 ]]; then
#     ensure_dataset() {
#         local repo="$1"
#         local destination="$2"
#         if [[ ! -d "$destination" || -z "$(find "$destination" -type f -print -quit)" ]]; then
#             hf download "$repo" --repo-type dataset --local-dir "$destination"
#         fi
#     }

#     mkdir -p -- "$EVAL_DATA_DIR/code_eval"
#     ensure_dataset openai/gsm8k "$EVAL_DATA_DIR/gsm8k"
#     ensure_dataset qintongli/GSM-Plus "$EVAL_DATA_DIR/gsm_plus"
#     ensure_dataset EleutherAI/hendrycks_math "$EVAL_DATA_DIR/hendrycks_math"
#     ensure_dataset google-research-datasets/mbpp "$EVAL_DATA_DIR/mbpp"
#     ensure_dataset allenai/sciq "$EVAL_DATA_DIR/sciq"
#     ensure_dataset cais/mmlu "$EVAL_DATA_DIR/mmlu"
#     ensure_dataset TIGER-Lab/MMLU-Pro "$EVAL_DATA_DIR/mmlu_pro"
#     ensure_dataset SaylorTwift/bbh "$EVAL_DATA_DIR/bbh"
#     if [[ ! -s "$EVAL_DATA_DIR/code_eval/code_eval.py" ]]; then
#         curl --fail --location \
#             https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/code_eval.py \
#             --output "$EVAL_DATA_DIR/code_eval/code_eval.py"
#     fi
#     if [[ ! -s "$EVAL_DATA_DIR/code_eval/execute.py" ]]; then
#         curl --fail --location \
#             https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/execute.py \
#             --output "$EVAL_DATA_DIR/code_eval/execute.py"
#     fi
# fi

# if [[ "$DRY_RUN" == 0 && ( ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ) ]]; then
#     PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}" \
#     python "$BASE_PATH/tools/process_data_ultraInteract.py" \
#         --base-path "$BASE_PATH" \
#         --data-dir "$RAW_DATA" \
#         --processed-data-dir "$PROCESSED_DATA_ROOT" \
#         --model-path "$CKPT" \
#         --model-type gemma \
#         --data-process-workers "${DATA_PROCESS_WORKERS:-32}" \
#         --max-length "$MAX_LENGTH" \
#         --max-prompt-length "$MAX_PROMPT_LENGTH" \
#         --dev-num "$DEV_NUM" \
#         --seed "$SEED"
# fi

# Pairwise adaptive probabilities: ON changes in on_self; SELF changes in off_self.
# The other enabled mode always receives the complementary probability.
export RHO_SELF_INIT="${RHO_SELF_INIT:-0.10}"
export RHO_ON_INIT="${RHO_ON_INIT:-0.05}"
export RHO_SELF_MAX="${RHO_SELF_MAX:-0.25}"
export RHO_ON_MAX="${RHO_ON_MAX:-0.25}"
export RHO_SELF_INCREMENT="${RHO_SELF_INCREMENT:-0.025}"
export RHO_ON_INCREMENT="${RHO_ON_INCREMENT:-0.025}"

if [[ "$DRY_RUN" == 0 && ( ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
fi

MODE_CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$MODE_CHECKPOINT_FILE"' EXIT
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/gemma-2-2b-it-distill/adaptive_mode_ablation}"
ENTRYPOINT_TAG="$(basename -- "${FINETUNE_ENTRYPOINT%.py}")"

for ADAPTIVE_MODE_SET in "${ADAPTIVE_ABLATIONS[@]}"; do
    MODE_KD_RATIO="$KD_RATIO"
    if [[ "$ADAPTIVE_MODE_SET" == self_only ]]; then
        MODE_KD_RATIO=1.0
    fi
    CONFIG_TAG="${ENTRYPOINT_TAG}_${KD_LOSS}_kd${MODE_KD_RATIO}_a${SKEW_ALPHA}_k${DISTILL_TOP_K}_t${DISTILL_TEMPERATURE}"
    CONFIG_TAG+="_geo${GEOMETRY}_mag${MAG_WEIGHT}_gram${GRAM_WEIGHT}_cka${CKA}_ckaw${CKA_WEIGHT}"
    CONFIG_TAG+="_menger${MENGER_WEIGHT}-eps${MENGER_EPS}_ctx${CONTEXT_MAX_NEW_TOKENS}-drop${SELF_DISTILL_CONTEXT_DROP_MAX}"
    case "$ADAPTIVE_MODE_SET" in
        on_self)
            SCHEDULE_TAG="rhoon${RHO_ON_INIT}-${RHO_ON_MAX}-${RHO_ON_INCREMENT}"
            ;;
        off_self)
            SCHEDULE_TAG="rhoself${RHO_SELF_INIT}-${RHO_SELF_MAX}-${RHO_SELF_INCREMENT}"
            ;;
        self_only)
            SCHEDULE_TAG="fixed_refresh${EVAL_INTERVAL}_seed${SEED}"
            ;;
    esac
    if [[ "$ADAPTIVE_MODE_SET" != self_only ]]; then
        SCHEDULE_TAG+="_threshold${ADAPTIVE_THRESHOLD}_eval${EVAL_INTERVAL}-seed${SELF_DISTILL_EVAL_SEED}_seed${SEED}"
    fi
    MODE_SAVE_PATH="$RESULTS_ROOT/$ADAPTIVE_MODE_SET/$CONFIG_TAG/$SCHEDULE_TAG"

    printf '\n[%s 1/2] Train Gemma ablation: %s\n' \
        "$ADAPTIVE_MODE_SET" "$ADAPTIVE_MODE_SET"
    : > "$MODE_CHECKPOINT_FILE"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" DATA_DIR="$DATA_DIR" \
        BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" VENV_PATH="$VENV_PATH" \
        CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
        PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" SAVE_PATH="$MODE_SAVE_PATH" \
        MAX_LENGTH="$MAX_LENGTH" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
        T_MAX_LENGTH="$T_MAX_LENGTH" T_MAX_PROMPT_LENGTH="$T_MAX_PROMPT_LENGTH" \
        CONTEXT_MAX_NEW_TOKENS="$CONTEXT_MAX_NEW_TOKENS" \
        DEV_NUM="$DEV_NUM" SEED="$SEED" \
        KD_LOSS="$KD_LOSS" KD_RATIO="$KD_RATIO" SKEW_ALPHA="$SKEW_ALPHA" \
        DISTILL_TOP_K="$DISTILL_TOP_K" DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
        GEOMETRY="$GEOMETRY" CKA="$CKA" MAG_WEIGHT="$MAG_WEIGHT" \
        GRAM_WEIGHT="$GRAM_WEIGHT" CKA_WEIGHT="$CKA_WEIGHT" \
        MENGER_WEIGHT="$MENGER_WEIGHT" MENGER_EPS="$MENGER_EPS" \
        SELF_DISTILL_CONTEXT_DROP_MAX="$SELF_DISTILL_CONTEXT_DROP_MAX" \
        EVAL_INTERVAL="$EVAL_INTERVAL" ADAPTIVE_THRESHOLD="$ADAPTIVE_THRESHOLD" \
        SELF_DISTILL_EVAL_SEED="$SELF_DISTILL_EVAL_SEED" \
        FINETUNE_ENTRYPOINT="$FINETUNE_ENTRYPOINT" \
        ADAPTIVE_MODE_SET="$ADAPTIVE_MODE_SET" \
        FINAL_CHECKPOINT_FILE="$MODE_CHECKPOINT_FILE" \
        bash "$BASE_PATH/scripts/gemma/train_gemma2_9b_to_2b.sh" "$@"

    if [[ "$DRY_RUN" == 1 ]]; then
        continue
    fi
    MODE_LORA_PATH="$(<"$MODE_CHECKPOINT_FILE")"
    [[ -f "$MODE_LORA_PATH/adapter_config.json" ]] || {
        printf 'Final LoRA checkpoint missing: %s\n' "$MODE_LORA_PATH" >&2
        exit 1
    }
    if [[ "$RUN_EVAL" == 0 ]]; then
        printf '\n[%s 2/2] Evaluation skipped (RUN_EVAL=0)\n' "$ADAPTIVE_MODE_SET"
        continue
    fi
    printf '\n[%s 2/2] Evaluate checkpoint: %s\n' \
        "$ADAPTIVE_MODE_SET" "$MODE_LORA_PATH"
    CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}" \
        LORA_PATH="$MODE_LORA_PATH" MODEL_PATH="$CKPT" \
        BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" \
        EVAL_VENV_PATH="$VENV_PATH" EVAL_PYTHON="$VENV_PATH/bin/python" \
        EVAL_DATA_DIR="$EVAL_DATA_DIR" \
        SAVE_PATH="$(dirname -- "$MODE_LORA_PATH")" \
        EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
        bash "$BASE_PATH/scripts/eval/eval_gemma.sh" run
done
