BASE_PATH=${1}

MAX_LENGTH=4096

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/minipile \
    --processed-data-dir ${BASE_PATH}/processed_data/minipile/codes/${MAX_LENGTH}/ \
    --model-path seeklhy/codes-1b-bird \
    --max-length ${MAX_LENGTH} \
    --train-num 990000 \
    --data-process-workers 16 \
    --dev-num 10000 \