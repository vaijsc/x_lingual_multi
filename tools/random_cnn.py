from datasets import load_dataset
import os
import re
import json

def input_template(article: str, input_prefix: str = "###\nArticle: ", input_suffix: str = "\n\n", output_command: str = "Summarize the above article in 3 sentences.") -> str:
    return f"{input_prefix}{article}{input_suffix}{output_command}"

dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
random_data = dataset.shuffle(seed=42).select(range(50_000))
random_data = random_data.train_test_split(test_size=0.02)
random_data['valid'] = random_data.pop('test')

os.makedirs("data/cnn/random", exist_ok=True)

num = 0
for split in ['train', 'valid']:
    with open(f"data/cnn/random/{split}.jsonl", "w+") as fout:
        for data in random_data[split]:
            inp = input_template(data['article'])
            out = data['highlights']
            json_data = {'prompt': inp, 'output': out}
            fout.write(json.dumps(json_data) + "\n")
            num += 1

print("Number of lines:", num)

test_data = load_dataset('cnn_dailymail', '3.0.0', split='test').shuffle(seed=42).select(range(5_000))
num = 0
with open(f"data/cnn/test.jsonl", "w+") as fout:
    for data in test_data:
        inp = input_template(data['article'])
        out = data['highlights']
        json_data = {'prompt': inp, 'output': out}
        fout.write(json.dumps(json_data) + "\n")
        num += 1

print("Number of lines:", num)
