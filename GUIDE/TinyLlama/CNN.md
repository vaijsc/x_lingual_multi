# Run CNN

## Get data

### LM training data

```bash
bash scripts/tiny_llama2/tools/process_data_minipile.sh .
```

### Random subset
```bash
python tools/random_cnn.py
```

Update the `scripts/tiny_llama2/tools/process_data_cnn.sh`

```bash
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
```

Then run it: `bash scripts/tiny_llama2/tools/process_data_cnn.sh .`

#### SFT method
```bash
bash scripts/tiny_llama2/sft/sft_lora_cnn_random.sh . 2012 4
```

#### MiniLLM method
```bash
bash scripts/tiny_llama2/init/init_lora_cnn_random . 2012 4
bash scripts/tiny_llama2/sft/sft_lora_cnn_random_teacher.sh . 2012 4
bash scripts/tiny_llama2/minillm/minillm_lora_cnn_random.sh . 2012 4
```

#### SeqKD method
```bash
bash scripts/tiny_llama2/tools/generate_data_seqkd.sh . 2013 2
```

Update the file `scripts/tiny_llama2/tools/process_pseudo_data_seqkd.sh`

```bash
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
```

Run
```bash
bash scripts/tiny_llama2/tools/process_pseudo_data_seqkd.sh .
```

Train
```bash
bash scripts/tiny_llama2/seqkd/seqkd_lora_cnn_random.sh . 2012 2
```

#### GKD method
```bash
bash scripts/tiny_llama2/gkd/gkd_lora_cnn_random.sh . 2012 4
```

#### Evaluation
```bash
bash scripts/tiny_llama2/eval/run_eval.sh . 0 path/to/lora-ckpt distinct-exp-name
```