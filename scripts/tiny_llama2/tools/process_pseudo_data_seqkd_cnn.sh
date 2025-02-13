BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_seqkd.py \
    --data-dir ${BASE_PATH}/results/llama2-7B/cnn-random/gen/stu/t1.0-l1024 \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/pseudo \
    --model-path meta-llama/Llama-2-7b-hf \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num -1 \
    --model-type tinyllama2

cp ${BASE_PATH}/processed_data/cnn-random/full/tinyllama2/valid_0.bin ${BASE_PATH}/processed_data/cnn-random/pseudo/tinyllama2/
cp ${BASE_PATH}/processed_data/cnn-random/full/tinyllama2/valid_0.idx ${BASE_PATH}/processed_data/cnn-random/pseudo/tinyllama2/
cp ${BASE_PATH}/processed_data/cnn-random/full/tinyllama2/valid.jsonl ${BASE_PATH}/processed_data/cnn-random/pseudo/tinyllama2/