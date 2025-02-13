# DistiLLM: Towards Streamlined Distillation for Large Language Models

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Train / Validation Data
```

## Base Pre-trained Models: GPT-2

## Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt` and `scripts/openllama2`. All our experiments are conducted on 4 \* 40A100, which can be reduced for small models.

## Modify the path to model checkpoint (Optional)
**NOTE**: If you copy the model checkpoint to the `checkpoint` folder like what the authors did, skip this step.

If you download the model from HuggingFace using normal python script. `AutoModelForCausalLM.from_pretrained(model_name)`, replace the code in file `scripts/gpt2/sft/sft_xlarge.sh`

```bash
CKPT_NAME="gpt2-xlarge"
# CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
CKPT="openai-community/gpt2-xl" # download automatically
```

### Baselines
The final checkpoints are selected by the **ROUGE-L** scores.

#### Student Initialization
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/init/init_base.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### MiniLLM Baselines

First need to download pre-trained data, here, we use small scale: openwebtext-10k,

```bash
python tools/get_openwebtext_small.py
# Remember to change the --model_path to point to the tokenizer/model checkpoint
bash scripts/gpt2/tools/process_data_pretrain_small.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Update the `scripts/gpt2/minillm/train_base_xl.sh` towards the correct `--data_dir` for pre-trained data

```bash
bash scripts/gpt2/minillm/train_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### GKD baselines

```bash
bash scripts/gpt2/gkd/gkd_base_xl.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

## Base Pre-trained Models: Llama2-7B

### Download model
```bash
huggingface-cli login # pass the Token
huggingface-cli whoami # output: transZ ==> great
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer 
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf', use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', use_auth_token=True)
```

## Modify the path to model checkpoint (Optional)
**NOTE**: If you copy the model checkpoint to the `checkpoint` folder like what the authors did, skip this step.

If you download the model from HuggingFace using normal python script. `AutoModelForCausalLM.from_pretrained(model_name)`, replace the code in file `scripts/gpt2/sft/sft_xlarge.sh`

```bash
CKPT_NAME="gpt2-xlarge"
# CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
CKPT="openai-community/gpt2-xl" # download automatically
```

File `scripts/llama2/tools/process_data_dolly.sh`
```bash
# The following use for HugginFace .cache, if you store the checkpoint at different place, just point to the folder contains (`.safetensors` files or `tokenizer_config.json`)
--model-path meta-llama/Llama-2-7b-hf \
```

## Preprocess

```bash
bash scripts/llama2/tools/process_data_dolly.sh /root/research/llm_kd/distillm/
```

## Train

### Baselines
The final checkpoints are selected by the **ROUGE-L** scores.

#### Student Initialization
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/llama2/init/init_7B_lora.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### Fine-tune the teacher models
```bash
bash scripts/llama2/sft/sft_13B_lora.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

#### MiniLLM Baselines

First need to download pre-trained data, here, we use small scale: openwebtext-10k,

```bash
python tools/get_openwebtext_small.py
# Remember to change the --model_path to point to the tokenizer/model checkpoint
bash scripts/llama2/tools/process_data_pretrain_small.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Update the `scripts/llama2/minillm/train_7B_13B_lora.sh` towards the correct `--data_dir` for pre-trained data

```bash
bash scripts/llama2/minillm/train_7B_13B_lora.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### GKD baselines

```bash
bash scripts/llama2/gkd/gkd_7B_13B_lora.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```