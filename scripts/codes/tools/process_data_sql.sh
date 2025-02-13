BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_sql.py \
    --data-dir ${BASE_PATH}/data/sql/v4 \
    --processed-data-dir ${BASE_PATH}/processed_data/sql-v4/prompt \
    --model-path seeklhy/codes-1b-bird \
    --data-process-workers 32 \
    --max-prompt-length 3850 \
    --only-prompt \
    --model-type codes

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_sql.py \
    --data-dir ${BASE_PATH}/data/sql/v4 \
    --processed-data-dir ${BASE_PATH}/processed_data/sql-v4/full \
    --model-path seeklhy/codes-1b-bird \
    --data-process-workers 32 \
    --max-prompt-length 3850 \
    --model-type codes
