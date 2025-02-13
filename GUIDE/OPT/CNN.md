# Run CNN

## Get data

### LM training data

```bash
bash scripts/opt/tools/process_data_minipile.sh .
```

### Random subset
```bash
python tools/random_cnn.py
```

Update the `scripts/opt/tools/process_data_cnn.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/prompt \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --only-prompt \
    --model-type opt

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/random \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/full \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --model-type opt
```

Then run it: `bash scripts/opt/tools/process_data_cnn.sh .`

#### SFT method
```bash
bash scripts/opt/sft/sft_lora_cnn_random.sh . 2012 4
```

#### MiniLLM method
```bash
bash scripts/opt/init/init_lora_cnn_random.sh . 2012 4
bash scripts/opt/sft/sft_lora_cnn_random_teacher.sh . 2012 4
bash scripts/opt/minillm/minillm_lora_cnn_random.sh . 2012 4
```

#### SeqKD method
```bash
bash scripts/opt/tools/generate_data_seqkd.sh . 2012 1
```

Update the file `scripts/opt/tools/process_pseudo_data_seqkd.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/results/opt-7b/cnn-random/gen/stu/t1.0-l1024 \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-random/pseudo \
    --model-path facebook/opt-1.3b \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --dev-num -1 \
    --model-type opt

cp ${BASE_PATH}/processed_data/cnn-random/full/opt/valid_0.bin ${BASE_PATH}/processed_data/cnn-random/pseudo/opt/
cp ${BASE_PATH}/processed_data/cnn-random/full/opt/valid_0.idx ${BASE_PATH}/processed_data/cnn-random/pseudo/opt/
cp ${BASE_PATH}/processed_data/cnn-random/full/opt/valid.jsonl ${BASE_PATH}/processed_data/cnn-random/pseudo/opt/
```

Run
```bash
bash scripts/opt/tools/process_pseudo_data_seqkd.sh . 
```

Training
```bash
bash scripts/opt/seqkd/seqkd_lora_cnn_random.sh . 2012 2
```

#### GKD method
```bash
bash scripts/opt/gkd/gkd_lora_cnn_random.sh . 2012 4
```

#### Evaluation
```bash
bash scripts/opt/eval/run_eval.sh . 0 path/to/lora-ckpt distinct-exp-name
```