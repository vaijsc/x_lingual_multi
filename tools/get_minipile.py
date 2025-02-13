from datasets import load_dataset
import os
import re

dataset = load_dataset('JeanKaddour/minipile', split='train')

os.makedirs("data/minipile", exist_ok=True)

num = 0
with open("data/minipile/data.txt", "w+") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)