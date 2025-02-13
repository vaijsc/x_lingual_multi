from datasets import load_dataset
import os
import re
import json
from typing import Dict, Tuple
import random

random.seed(42)

def construct_template(dict: Dict) -> Tuple[str, str]:
    # construct prompt
    template = f"Passage: {dict['support']}\nQuestion: {dict['question']}\n"
    
    answers = [dict['distractor1'], dict['distractor2'], dict['distractor3'], dict['correct_answer']]
    random.shuffle(answers)

    answer_choices = ['A', 'B', 'C', 'D']
    multiple_choices = [f'{choice}. {answer}' for choice, answer in zip(answer_choices, answers)]
    multiple_choices_str = "\n".join(multiple_choices)
    template += multiple_choices_str
    template += "\nAnswer: "

    for idx, answer in enumerate(answers):
        if answer == dict['correct_answer']:
            choice = answer_choices[idx] + ". " + dict['correct_answer']

    return template, choice

# def construct_template(dict: Dict) -> Tuple[str, str]:
#     # construct prompt
#     template = f"Passage: {dict['support']}\nQuestion: {dict['question']}\nAnswer: "
#     choice = dict['correct_answer']
    
#     return template, choice

dataset = load_dataset('allenai/sciq')

os.makedirs("data/sciq", exist_ok=True)

num = 0
for split in ['train', 'validation', 'test']:
    with open(f"data/sciq/{split}.jsonl", "w+") as fout:
        for data in dataset[split]:
            inp, out = construct_template(data)
            json_data = {'prompt': inp, 'output': out}
            fout.write(json.dumps(json_data) + "\n")
            if split in ['train', 'validation']:
                num += 1

print("Number of lines:", num)