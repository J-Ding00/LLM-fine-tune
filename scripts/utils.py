import os
import json
import yaml
from collections import defaultdict
from pathlib import Path

def summarize_split_stats(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    traits = config["data_split"]["stratified_criteria"]
    threshold = config["data_split"]["threshold"]
    prefix = Path(config["data_split"]["input"]).stem
    base_dir = Path(config["data_split"]["output_dir"])

    def get_score_bin(score):
        return "high" if score > threshold else "low"

    for split in ["train", "val", "test"]:
        path = base_dir / split / f"{prefix}_{split}.jsonl"
        with open(path, "r") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        trait_counts = {trait: defaultdict(int) for trait in traits}

        for sample in lines:
            try:
                label = json.loads(sample["label"])
                for trait in traits:
                    if trait in label:
                        bin_key = get_score_bin(label[trait]["score"])
                        trait_counts[trait][bin_key] += 1
            except Exception as e:
                print(f"Skipping malformed sample: {e}")

        print(f"\nStats for {split.capitalize()}: {len(lines)} samples")
        for trait in traits:
            total = sum(trait_counts[trait].values())
            low = trait_counts[trait]["low"]
            high = trait_counts[trait]["high"]
            print(f"  Trait: {trait.ljust(18)} | low: {low:<4} ({low/total:.1%}) | high: {high:<4} ({high/total:.1%})")
    

def print_folder_structure(root_path):
    exclude_dirs = {'venv', '.git'}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Modify dirnames in-place to skip excluded folders
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        print(f"\n📁 {dirpath}")
        for file in filenames:
            print(f"  📄 {file}")

def average_transcript_length(jsonl_path):
    total_words = 0
    total_samples = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                word_count = len(text.strip().split())
                total_words += word_count
                total_samples += 1
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    if total_samples == 0:
        return 0

    avg_length = total_words / total_samples
    print(f"Average transcript length: {avg_length:.2f} words ({total_samples} samples)")
    return avg_length

if __name__ == "__main__":
    # print_folder_structure(".")
    # average_transcript_length('data/raw/openwebtext_sample.jsonl')
    summarize_split_stats("config.yaml")
    exit()