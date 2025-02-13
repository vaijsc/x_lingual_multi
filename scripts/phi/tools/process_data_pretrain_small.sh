BASE_PATH=${1}

MAX_LENGTH=1024

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext-10k \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/phi/${MAX_LENGTH}/ \
    --model-path microsoft/phi-1_5 \
    --max-length ${MAX_LENGTH} \
    --train-num 9000 \
    --data-process-workers 32 \
    --dev-num 1000 \