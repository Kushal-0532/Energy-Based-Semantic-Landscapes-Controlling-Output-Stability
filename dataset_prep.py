import json
import random
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

stsb = load_dataset("sentence-transformers/stsb", split="train")
snli = load_dataset("snli", split="train").filter(lambda x: x["label"] != -1)

positives = [{"sentence1": row["sentence1"], "sentence2": row["sentence2"]} for row in stsb if row["score"] >= 4.0]
negatives_easy = [{"sentence1": row["sentence1"], "sentence2": row["sentence2"]} for row in stsb if row["score"] <= 1.0]
negatives_hard = [{"sentence1": row["premise"], "sentence2": row["hypothesis"]} for row in snli if row["label"] == 2]

negatives_shuffled = []
for pair in positives:
    words = pair["sentence1"].split()
    random.shuffle(words)
    negatives_shuffled.append({"sentence1": " ".join(words), "sentence2": pair["sentence2"]})

for filename, data in [
    ("data/positives.json", positives),
    ("data/negatives_easy.json", negatives_easy),
    ("data/negatives_hard.json", negatives_hard),
    ("data/negatives_shuffled.json", negatives_shuffled),
]:
    with open(filename, "w") as f:
        json.dump(data, f)

print(f"positives: {len(positives)}")
print(f"negatives_easy: {len(negatives_easy)}")
print(f"negatives_hard: {len(negatives_hard)}")
print(f"negatives_shuffled: {len(negatives_shuffled)}")
