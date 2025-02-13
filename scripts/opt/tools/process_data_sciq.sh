BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/sciq \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/prompt \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --only-prompt \
    --model-type opt

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/sciq \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/full \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --model-type opt
