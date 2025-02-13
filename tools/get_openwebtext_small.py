import datasets
import os
import re

dataset = datasets.load_dataset('stas/openwebtext-10k', split='train')

os.makedirs("data/openwebtext-10k", exist_ok=True)

num = 0
with open("data/openwebtext-10k/data.txt", "w+") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)