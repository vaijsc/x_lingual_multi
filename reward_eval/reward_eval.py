from transformers import AutoTokenizer, AutoModel
import argparse
import pandas as pd
from typing import List, Dict
import json
from loguru import logger
import os
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--ckpt', type=str, default='openbmb/Eurus-RM-7b')

    # data input
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--prompt_field', type=str, default='prompt')
    parser.add_argument('--answer1_path', type=str)
    parser.add_argument('--answer2_path', type=str)
    parser.add_argument('--answer_field', type=str, default='text')
    parser.add_argument('--data_name', type=str)

    # cache and output
    parser.add_argument('--model_1_type_id', type=str)
    parser.add_argument('--model_2_type_id', type=str)
    parser.add_argument('--load_data_answer_1', type=str, default=None)
    parser.add_argument('--load_data_answer_2', type=str, default=None)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args

def load_tokenizer(ckpt: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    return tokenizer

def load_model(ckpt: str) -> AutoModel:
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
    return model

def emit_reward(model: AutoModel, tokenizer: AutoTokenizer, inputs: str) -> float:
    inp_tensor = tokenizer(inputs, return_tensors="pt")
    inp_tensor = {k: v.to(model.device) for k, v in inp_tensor.items()}
    try:
        result = model(**inp_tensor).item()
        return result
    except:
        # CUDA OOM
        return -1e5

def load_array(path: str):
    df = pd.read_csv(path)
    item = df['value'].values
    return item

def store_array(item: List[float], at: str):
    ext = '.csv'
    filename = f'{at}{ext}'
    df = pd.DataFrame({'value': item})
    df.to_csv(filename)

def load_data(args: argparse.ArgumentParser) -> List[Dict[str, str]]:
    # read JSONL file
    prompts = []
    with open(args.prompt_path) as fin:
        for line in fin:
            data = json.loads(line)
            prompts.append(data[args.prompt_field])
    
    answer1 = []
    with open(args.answer1_path) as fin:
        for line in fin:
            data = json.loads(line)
            answer1.append(data[args.answer_field])

    answer2 = []
    with open(args.answer2_path) as fin:
        for line in fin:
            data = json.loads(line)
            answer2.append(data[args.answer_field])
    
    data = [{'prompt': p, 'answer1': a1, 'answer2': a2} for p, a1, a2 in zip(prompts, answer1, answer2)]
    return data

if __name__ == "__main__":
    args = get_args()

    logger.info(f"Testing on {args.data_name} dataset.")

    tokenizer = load_tokenizer(args.ckpt)
    model = load_model(args.ckpt)
    device = torch.device('cuda:0')
    model = model.to(device)
    logger.success('Loading tokenizer and model successfully')
    data = load_data(args) # [{prompt, answer1, answer2}]
    
    logger.success('Loading data successfully')
    if args.load_data_answer_1 is not None:
        answer1_rewards = load_array(args.load_data_answer_1)
    else:
        answer1_rewards = []

    if args.load_data_answer_2 is not None:
        answer2_rewards = load_array(args.load_data_answer_2)
    else:
        answer2_rewards = []

    diffs = []
    len_10_percent = int(len(data) * 0.1)
    cnt = 0
    for i, d in enumerate(data):
        if (i+1) % len_10_percent == 0:
            cnt += 10
            logger.info(f'Processing reach {cnt}% of data')
        if args.load_data_answer_1 is None:
            answer1_reward = emit_reward(model, tokenizer, d['prompt'] + d['answer1'])
            answer1_rewards.append(answer1_reward)
        else:
            answer1_reward = answer1_rewards[i]

        if args.load_data_answer_2 is None:
            answer2_reward = emit_reward(model, tokenizer, d['prompt'] + d['answer2'])
            answer2_rewards.append(answer2_reward)
        else:
            answer2_reward = answer2_rewards[i]
    
        diffs.append(answer1_reward - answer2_reward)
    
    logger.info('Storing ...')
    os.makedirs(args.output_path, exist_ok=True)
    if args.load_data_answer_1 is None:
        store_array(item=answer1_rewards, at=f'{args.output_path}/{args.model_1_type_id}')
    
    if args.load_data_answer_2 is None:
        store_array(item=answer2_rewards, at=f'{args.output_path}/{args.model_2_type_id}')

    store_array(item=diffs, at=f'{args.output_path}/{args.model_1_type_id}-{args.model_2_type_id}')
    logger.success('Stored successfully')
    wins = sum([1 for diff in diffs if diff > 0])
    logger.info(f'Win rate: {(wins/len(data)):.2f}, Lose rate: {(1 - wins/len(data)):.2f}')