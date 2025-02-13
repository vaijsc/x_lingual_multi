BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_alpaca.py \
    --data-dir ${BASE_PATH}/data/alpaca/ \
    --processed-data-dir ${BASE_PATH}/processed_data/alpaca/prompt \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type tinyllama2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_alpaca.py \
    --data-dir ${BASE_PATH}/data/alpaca/ \
    --processed-data-dir ${BASE_PATH}/processed_data/alpaca/full \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type tinyllama2
