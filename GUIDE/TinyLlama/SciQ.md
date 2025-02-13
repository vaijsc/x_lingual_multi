# Run SciQ

## Get data

### LM training data

```bash
bash scripts/tiny_llama2/tools/process_data_minipile.sh .
```

### Random subset
```bash
python tools/sciq.py
```

Update the `scripts/tiny_llama2/tools/process_data_sciq.sh`

```bash
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
```

Then run it: `bash scripts/tiny_llama2/tools/process_data_sciq.sh .`

#### SFT method
```bash
bash scripts/tiny_llama2/sft/sft_lora_sciq.sh . 2012 4
```

#### MiniLLM method
```bash
bash scripts/tiny_llama2/init/init_lora_sciq.sh . 2012 4
bash scripts/tiny_llama2/sft/sft_lora_sciq_teacher.sh . 2012 4
bash scripts/tiny_llama2/minillm/minillm_lora_sciq.sh . 2012 4
```

#### SeqKD method
```bash
bash scripts/tiny_llama2/tools/generate_data_seqkd_sciq.sh . 2012 2
```

Update the file `scripts/tiny_llama2/tools/process_pseudo_data_seqkd_sciq.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_seqkd.py \
    --data-dir ${BASE_PATH}/results/llama2-7B/sciq/gen/stu/t1.0-l300 \
    --processed-data-dir ${BASE_PATH}/processed_data/sciq/pseudo \
    --model-path meta-llama/Llama-2-7b-hf \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type tinyllama2

cp ${BASE_PATH}/processed_data/sciq/full/tinyllama2/valid_0.bin ${BASE_PATH}/processed_data/sciq/pseudo/tinyllama2/
cp ${BASE_PATH}/processed_data/sciq/full/tinyllama2/valid_0.idx ${BASE_PATH}/processed_data/sciq/pseudo/tinyllama2/
cp ${BASE_PATH}/processed_data/sciq/full/tinyllama2/valid.jsonl ${BASE_PATH}/processed_data/sciq/pseudo/tinyllama2/
```

Run
```bash
bash scripts/tiny_llama2/tools/process_pseudo_data_seqkd_sciq.sh .
```

Train
```bash
bash scripts/tiny_llama2/seqkd/seqkd_lora_sciq.sh . 2012 2
```

#### GKD method
```bash
bash scripts/tiny_llama2/gkd/gkd_lora_sciq.sh . 2012 4
```

#### Evaluation
```bash
bash scripts/tiny_llama2/eval/run_eval.sh . 0 path/to/lora-ckpt distinct-exp-name
```