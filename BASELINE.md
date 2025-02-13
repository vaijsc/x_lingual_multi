# Baseline

- Student: TinyLlama2
- Teacher: Llama7b, 13b
- Data SFT: Alpaca
- Data LM for MiniLLM: OpenWebText-10K (hope to expand to 100K)

## Prepare model and data

**Download model**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
```

**Prepare data**
```bash
python tools/get_alpaca.py
bash scripts/tiny_llama2/tools/process_data_alpaca.sh ${/PATH/TO/DistiLLM}
bash scripts/tiny_llama2/tools/process_data_pretrain_small.sh ${/PATH/TO/DistiLLM}
```

## Training

**NOTE**: BATCH_SIZE * GRAD_ACC = 8, decrease BATCH_SIZE must increase GRAD_ACC

### Init student: SFT

```bash
bash scripts/tiny_llama2/init/init_1.1B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### SFT Teacher

Llama2-7B

```bash
bash scripts/llama2/sft/sft_7B_lora_alpaca.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Llama2-13B

```bash
bash scripts/llama2/sft/sft_13B_lora_alpaca.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### SeqKD

Llama2-7B

```bash
bash scripts/tiny_llama2/tools/generate_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/tiny_llama2/tools/process_pseudo_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/tiny_llama2/seqkd/seqkd_1.1B_7B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Llama2-13B

```bash
bash scripts/tiny_llama2/tools/generate_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/tiny_llama2/tools/process_pseudo_data_seqkd.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
bash scripts/tiny_llama2/seqkd/seqkd_1.1B_13B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### MiniLLM

Llama2-7B

```bash
bash scripts/tiny_llama2/minillm/minillm_1.1B_7B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Llama2-13B

```bash
bash scripts/tiny_llama2/minillm/minillm_1.1B_13B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

### GKD

Llama2-7B

```bash
bash scripts/tiny_llama2/gkd/gkd_1.1B_7B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```

Llama2-13B

```bash
bash scripts/tiny_llama2/gkd/gkd_1.1B_13B.sh ${/PATH/TO/DistiLLM} ${MASTER_PORT} ${GPU_NUM}
```