import json
import random
import os
from pathlib import Path
from collections import defaultdict
import yaml

def get_score_bin(score, threshold):
    return "high" if score > threshold else "low"

def get_stratification_key(example, traits, threshold):
    try:
        label = json.loads(example['label'])
        bins = []
        for trait in traits:
            if trait in label:
                bins.append(get_score_bin(label[trait]["score"], threshold))
        return "_".join(bins)
    except Exception:
        return "unknown"

def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def split_jsonl_stratified(input_path, traits, threshold, train_ratio=0.8, val_ratio=0.1, seed=42, output_dir="data"):
    with open(input_path, "r") as f:
        raw_lines = [json.loads(line.strip()) for line in f if line.strip()]

    random.seed(seed)
    stratified_groups = defaultdict(list)

    for ex in raw_lines:
        key = get_stratification_key(ex, traits, threshold)
        stratified_groups[key].append(ex)

    train_set, val_set, test_set = [], [], []

    for key, group in stratified_groups.items():
        random.shuffle(group)
        total = len(group)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_set.extend(group[:train_end])
        val_set.extend(group[train_end:val_end])
        test_set.extend(group[val_end:])

    print(f"Stratified by traits: {traits} with global threshold: {threshold}")
    print(f"Total samples: {len(raw_lines)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    input_name = Path(input_path).stem
    write_jsonl(train_set, f"{output_dir}/train/{input_name}_train.jsonl")
    write_jsonl(val_set, f"{output_dir}/val/{input_name}_val.jsonl")
    write_jsonl(test_set, f"{output_dir}/test/{input_name}_test.jsonl")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_path = config["data_split"]["input"]
    output_dir = config["data_split"]["output_dir"]
    traits = config["data_split"]["traits"]
    threshold = config["data_split"]["threshold"]
    train_ratio = config["data_split"].get("train_ratio", 0.8)
    val_ratio = config["data_split"].get("val_ratio", 0.1)
    seed = config["data_split"].get("seed", 42)

    split_jsonl_stratified(
        input_path=input_path,
        traits=traits,
        threshold=threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        output_dir=output_dir
    )