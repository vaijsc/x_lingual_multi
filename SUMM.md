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

**NOTE**: full fine-tuning experiment

```bash
bash scripts/tiny_llama2/minillm/minillm_cnn_random.sh . 2012 4
```

Update the scripts for `train.sh`:

```bash
bash scripts/tiny_llama2/init/init_lora_cnn_random . 2012 4
bash scripts/tiny_llama2/sft/sft_lora_cnn_random.sh . 2012 4
bash scripts/tiny_llama2/sft/sft_lora_cnn_random_teacher.sh . 2012 4
```

### Cluster subset
**This require GPU.**
```bash
python tools/cluster_cnn.py
```

Update the `scripts/tiny_llama2/tools/process_data_cnn.sh`

```bash
BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/cluster \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-cluster/prompt \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --only-prompt \
    --model-type tinyllama2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_cnn.py \
    --data-dir ${BASE_PATH}/data/cnn/cluster \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn-cluster/full \
    --model-path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --data-process-workers 32 \
    --max-prompt-length 512 \
    --model-type tinyllama2
```

Then run it: `bash scripts/tiny_llama2/tools/process_data_cnn.sh .`

Update the scripts for `train_cluster.sh`:

```bash
bash scripts/tiny_llama2/init/init_lora_cnn_cluster.sh . 2012 2
bash scripts/tiny_llama2/sft/sft_lora_cnn_cluster.sh . 2012 2
bash scripts/tiny_llama2/sft/sft_lora_cnn_cluster_teacher.sh . 2012 2
```

### Evaluation
