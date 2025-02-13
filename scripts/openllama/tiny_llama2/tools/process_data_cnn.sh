BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
#     --data-dir ${BASE_PATH}/data/revise_cnn \
#     --processed-data-dir ${BASE_PATH}/processed_data/revise-cnn/prompt \
#     --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --data-process-workers 32 \
#     --max-prompt-length 1024 \
#     --only-prompt \
#     --model-type tinyllama2

# prompt and response for baselines
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
#     --data-dir ${BASE_PATH}/data/cnn/feedback-random/v123 \
#     --processed-data-dir ${BASE_PATH}/processed_data/cnn-fb-v123/full \
#     --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --data-process-workers 32 \
#     --max-prompt-length 700 \
#     --model-type tinyllama2

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
#     --data-dir ${BASE_PATH}/data/cnn/feedback-random/strategy_v2 \
#     --processed-data-dir ${BASE_PATH}/processed_data/cnn-strategy-v2/full \
#     --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --data-process-workers 32 \
#     --max-prompt-length 700 \
#     --model-type tinyllama2

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
#     --data-dir ${BASE_PATH}/data/cnn/feedback-random/v23 \
#     --processed-data-dir ${BASE_PATH}/processed_data/cnn-fb-v23/full \
#     --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
#     --data-process-workers 32 \
#     --max-prompt-length 700 \
#     --model-type tinyllama2

BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/prompt \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --only-prompt \
    --model-type tinyllama2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/full \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --model-type tinyllama2