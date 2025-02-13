#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT="facebook/opt-6.7b"
PEFT_CKPT_NAME="opt-7b"
PEFT_CKPT="${BASE_PATH}/results/opt/train/cnn-random/teacher-sft/e5-bs1-lr0.0005-G8-N2-NN1-lora-16-64-0.1/15310"
# data
DATA_DIR="${BASE_PATH}/processed_data/cnn-random/full/opt/"
# hp
EVAL_BATCH_SIZE=4
# runtime
SAVE_PATH="${BASE_PATH}/results/${PEFT_CKPT_NAME}/gen/"
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type opt"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names cnn"
OPTS+=" --num-workers 0"
OPTS+=" --gen-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type gen"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --bf16"


export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/generate.py ${OPTS} $@"


echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
