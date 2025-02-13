BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/sciq \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/prompt \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --only-prompt \
    --model-type tinyllama2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/sciq \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/full \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --model-type tinyllama2