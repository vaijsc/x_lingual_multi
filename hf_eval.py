from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import torch
import json, re
from loguru import logger
from tqdm import tqdm

JSONL_TEST_FILE="data/sql/v3/test.jsonl"
TOKENIZER_PATH="seeklhy/codes-15b-spider"
MODEL_PATH="seeklhy/codes-15b-spider"
LORA_PATH="results/codes/train/teacher-sft/e1-bs1-lr0.0005-G8-N1-NN1-lora-16-64-0.1/8186"
MAX_PROMPT_LENGTH=3850
MAX_LENGTH=4096
NEW_TOKENS=MAX_LENGTH - MAX_PROMPT_LENGTH
FILE_STORE="sql_teacher_sft_v2.jsonl"

if __name__ == "__main__":
    # read data
    data = []
    with open(JSONL_TEST_FILE) as fin:
        for line in fin:
            line = line[:-1] # exclude the last \n line
            _data = json.loads(line)
            data.append(_data)

    logger.info(f"Sample {len(data)} samples.")

    # loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if LORA_PATH is not None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", quantization_config=bnb_config)
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", load_in_8bit=True)
    device = model.device
    logger.success(f"Successfully load {MODEL_PATH} tokenizer and model.")

    # into the loop of tokenizer, generate, accumulate the result
    preds = []
    for _data in tqdm(data, desc="Evaluating"):
        text = _data['prompt']
        inputs = tokenizer(text, max_length=MAX_PROMPT_LENGTH, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=NEW_TOKENS)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(out)

    # write to file
    with open(FILE_STORE, "w+", encoding="utf-8") as fout:
        for d, p in zip(data, preds):
            write_json = {'prompt': d['prompt'], 'predict': p, 'label': d['output']}
            fout.write(json.dumps(write_json) + "\n")