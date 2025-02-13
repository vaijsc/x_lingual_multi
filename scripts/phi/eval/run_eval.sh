#!/bin/bash

MASTER_PORT=2040
BASE_PATH=${1}
DEVICE=${2}
ckpt=${3}
EXP_NAME=$4


CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/phi/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME 
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/phi/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/phi/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME
# CUDA_VISIBLE_DEVICES=${DEVICE} bash ${BASE_PATH}/scripts/phi/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 ${ckpt} $EXP_NAME