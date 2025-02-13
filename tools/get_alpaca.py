import datasets
import os, json
import re

dataset = datasets.load_dataset('tatsu-lab/alpaca', split='train')

os.makedirs("data/alpaca", exist_ok=True)

num = 0
with open("data/alpaca/raw.jsonl", "w+") as f:
    for data in dataset:
        f.write(json.dumps(data) + "\n")
        num += 1

print("Number of lines:", num)