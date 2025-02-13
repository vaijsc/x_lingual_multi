BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_seqkd.py \
    --data-dir ${BASE_PATH}/results/opt-7b/sciq/gen/stu/t1.0-l300 \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/pseudo \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type opt

cp ${BASE_PATH}/processed_data/sciq/full/opt/valid_0.bin ${BASE_PATH}/processed_data/sciq/pseudo/opt/
cp ${BASE_PATH}/processed_data/sciq/full/opt/valid_0.idx ${BASE_PATH}/processed_data/sciq/pseudo/opt/
cp ${BASE_PATH}/processed_data/sciq/full/opt/valid.jsonl ${BASE_PATH}/processed_data/sciq/pseudo/opt/