from transformers import AutoTokenizer
import json
from loguru import logger
from statistics import median, mean
from typing import Optional, List
from numpy import quantile

def apply_template_prompt(inst: str, inp: Optional[str]) -> str:
    if input is None or len(inp) == 0:
        template = (
            "<|im_start|>Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(instruction=inst)
    else:
        template = (
            "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
        )
        prompt = template.format(instruction=inst, input=inp)
    return prompt

def apply_template_full(inst: str, inp: Optional[str], output: str) -> str:
    prompt = apply_template_prompt(inst, inp)
    full = prompt + output
    return full

def calc_stats(ls: List[int]):
    print(f'Median: {median(ls)}')
    print(f'Mean: {mean(ls)}')
    print(f'Max: {max(ls)}')
    print(f'Min: {min(ls)}')
    print(f'1st Quantile: {quantile(ls, 0.25)}')
    print(f'3rd Quantile: {quantile(ls, 0.75)}')

if __name__ == "__main__":
    data = []
    with open('data/alpaca/raw.jsonl', 'r') as fin:
        for line in fin:
            _data = json.loads(line)
            data.append(_data)

    logger.info(f"Len of data: {len(data)}")
    # prompt
    prompts = [apply_template_prompt(d['instruction'], d['input']) for d in data]
    # prompts = [d['prompt'] for d in data]
    logger.info(f'Prompts sample: {prompts[0]}')
    # full
    full = [apply_template_full(d['instruction'], d['input'], d['output']) for d in data]
    # full = [d['prompt'] + d['output'] for d in data]
    logger.info(f'Prompts sample: {full[0]}')
    
    logger.info(f'Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    prompts_len = [len(tokenizer.encode(p)) for p in prompts]

    logger.info(f'Statistics of prompt length')
    calc_stats(prompts_len)

    fulls_len = [len(tokenizer.encode(f)) for f in full]

    logger.info(f'Statistics of full length')
    calc_stats(fulls_len)
    
    """
    2024-04-07 11:29:35.483 | INFO     | __main__:<module>:46 - Statistics of prompt length
    Median: 68.0
    Mean: 75.68201223029884
    Max: 573
    Min: 56
    1st Quantile: 63.0
    3rd Quantile: 86.0
    """

    """
    2024-04-07 11:36:07.072 | INFO     | __main__:<module>:58 - Statistics of full length
    Median: 118.0
    Mean: 132.5175185569786
    Max: 1066
    Min: 58
    1st Quantile: 93.0
    3rd Quantile: 158.0
    """

    """
    2024-06-07 11:05:36.965 | INFO     | __main__:<module>:60 - Statistics of prompt length
    Median: 78.0
    Mean: 86.60876504749817
    Max: 681
    Min: 64
    1st Quantile: 72.0
    3rd Quantile: 98.0
    2024-06-07 11:05:50.639 | INFO     | __main__:<module>:65 - Statistics of full length
    Median: 134.5
    Mean: 151.14059074650976
    Max: 1328
    Min: 66
    1st Quantile: 106.0
    3rd Quantile: 179.0
    """