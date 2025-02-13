#!/bin/bash

MASTER_PORT=2041
BASE_PATH=${1}
DEVICE=${2}
ckpt=${3}
EXP_NAME=$4

# instruction fine-tuning
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/tiny_llama2/eval/eval_main_self_inst.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/tiny_llama2/eval/eval_main_vicuna.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/tiny_llama2/eval/eval_main_sinst.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/tiny_llama2/eval/eval_main_uinst.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME

# cnn
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/tiny_llama2/eval/eval_main_cnn.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/opt/eval/eval_main_lora_cnn.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME

# sciq
CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/opt/eval/eval_main_lora_sciq.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
