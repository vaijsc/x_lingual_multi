#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/MiniLLM"}
# CKPT_NAME="openllama2-3B"
# CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
CKPT="${BASE_PATH}/results/tinyllama2/train/alpaca/init/tinyllama2/e3-bs8-lr5e-05-G1-N2-NN1/9561" # directory point to init checkpoint
# PEFT_CKPT_NAME="llama2-7B"
# PEFT_CKPT="${BASE_PATH}/results/llama2/train/minillm_init/${PEFT_CKPT_NAME}/"
TEACHER_CKPT_NAME="llama2-7B"
# TEACHER_CKPT="${BASE_PATH}/checkpoints/${TEACHER_CKPT_NAME}/"
TEACHER_CKPT="meta-llama/Llama-2-7b-hf"
TEACHER_PEFT_CKPT_NAME="llama2-7B"
TEACHER_PEFT_CKPT="${BASE_PATH}/results/llama2/train/alpaca/sft/${TEACHER_PEFT_CKPT_NAME}/e10-bs8-lr0.0005-G1-N2-NN1-lora-16-64-0.1/31870"
# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/alpaca/prompt/tinyllama2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/tiny_llama2/512/9K/"
# runtime
SAVE_PATH="${BASE_PATH}/results/tinyllama2/train/alpaca/minillm/tinyllama2-7B"
# hp
GRAD_ACC=4
BATCH_SIZE=2
CHUNK_SIZE=8



OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
OPTS+=" --scheduler-name cosine_trm"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 500"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# lora
# OPTS+=" --peft lora"
OPTS+=" --do-train"
# OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
# OPTS+=" --peft-path ${PEFT_CKPT}"
OPTS+=" --peft-teacher lora"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 256"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --bf16"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
