set -ex


export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

PROMPT_TYPE="qwen25-math-cot-extra"

BASE_MODEL_PATH="/home/sunl/ckpts/Qwen_Qwen2.5-Math-7B"      

LOSS_TYPE="balance_hinge_alpha_0.3"

LORA_PATH="/home/sunl/llm_rl/runs/dpo-qwen-2-5-7b-math-ep2-${LOSS_TYPE}"

MERGE_MODEL_PATH="/home/sunl/llm_rl/runs/dpo-qwen-2-5-7b-math-ep2-${LOSS_TYPE}-merge"

OUTPUT_DIR="runs/dpo-qwen-2-5-7b-math-ep2-${LOSS_TYPE}"

MAX_K=1
DATA_NAME="math500,olympiadbench,minerva_math,aime24,amc23"
SPLIT="test"

NUM_TEST_SAMPLE=-1
SEED=0
TEMPERATURE=0
START=0
END=-1
MAX_TOKENS=2048

# merge lora weight
python -u "evaluation/merge_adapter_weights.py" \
    --peft_model_id ${LORA_PATH} \
    --output_dir ${MERGE_MODEL_PATH} \
    --save_tokenizer True

# evaluate dataset
TOKENIZERS_PARALLELISM=false \
python -u "evaluation/math_eval.py" \
    --model_name_or_path ${MERGE_MODEL_PATH} \
    --max_tokens_per_call ${MAX_TOKENS} \
    --data_name ${DATA_NAME} \
    --data_dir "evaluation/data" \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --top_p 1 \
    --n_sampling ${MAX_K} \
    --start ${START} \
    --end ${END} \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --gpu_memory_utilization 0.9 \

rm -rf ${MERGE_MODEL_PATH}

