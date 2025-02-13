# Run CNN

## Get data

### LM training data

```bash
bash scripts/openllama/tools/process_data_minipile.sh .
```

### Random subset
```bash
python tools/random_cnn.py
```

Update the `scripts/openllama/tools/process_data_cnn.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/prompt \
    --model-path openlm-research/open_llama_7b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --only-prompt \
    --model-type openllama

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/full \
    --model-path openlm-research/open_llama_7b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --model-type openllama
```

Then run it: `bash scripts/openllama/tools/process_data_cnn.sh .`

#### SFT method
```bash
bash scripts/openllama/sft/sft_lora_cnn_random.sh . 2012 4
```

#### MiniLLM method
```bash
bash scripts/openllama/init/init_lora_cnn_random.sh . 2012 4
bash scripts/openllama/sft/sft_lora_cnn_random_teacher.sh . 2012 4
bash scripts/openllama/minillm/minillm_lora_cnn_random.sh . 2012 4
```

#### SeqKD method
```bash
bash scripts/openllama/tools/generate_data_seqkd_cnn.sh . 2013 1
```

Update the file `scripts/openllama/tools/process_pseudo_data_seqkd_cnn.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_seqkd.py \
    --data-dir ${BASE_PATH}/results/openllama2-7B/cnn-random/gen/stu/t1.0-l1024 \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/pseudo \
    --model-path openlm-research/open_llama_7b_v2 \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num -1 \
    --model-type openllama

cp ${BASE_PATH}/processed_data/cnn-random/full/openllama/valid_0.bin ${BASE_PATH}/processed_data/cnn-random/pseudo/openllama/
cp ${BASE_PATH}/processed_data/cnn-random/full/openllama/valid_0.idx ${BASE_PATH}/processed_data/cnn-random/pseudo/openllama/
cp ${BASE_PATH}/processed_data/cnn-random/full/openllama/valid.jsonl ${BASE_PATH}/processed_data/cnn-random/pseudo/openllama/
```

Run
```bash
bash scripts/openllama/tools/process_pseudo_data_seqkd_cnn.sh .
```

Train
```bash
bash scripts/openllama/seqkd/seqkd_lora_cnn_random.sh . 2012 2
```

#### GKD method
```bash
bash scripts/openllama/gkd/gkd_lora_cnn_random.sh . 2012 4
```

#### Evaluation
```bash
bash scripts/opt/eval/run_eval.sh . 0 path/to/lora-ckpt distinct-exp-name
```