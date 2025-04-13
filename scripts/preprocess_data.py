import json
import yaml
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def trim_transcript(transcript, min_len, max_len):
    words = transcript.strip().split()
    if len(words) < min_len or len(words) > max_len:
        return None
    return " ".join(words[:max_len])  # truncate if needed

def build_input_output(sample, criteria, criteria_format):
    transcript = sample["text"]
    label = json.loads(sample["label"])

    # You can further template this if needed
    
    input_text = f"""
    You are an expert speech evaluator. Given a transcript of a speaker's message, assess the communication quality in terms of the following traits:
    {criteria}

    For each trait, assign a score from 1 (poor) to 10 (excellent). Be honest and critical in your assessment without bias toward high scores. 
    Then provide a short explanation (1–2 sentences) justifying your score for that trait.
    
    The final output must be a JSON string with the exact format below:
    {criteria_format}
    Transcript:
    {transcript}
    """

    output_text = json.dumps(label, ensure_ascii=False)
    return input_text.strip(), output_text.strip()

def preprocess_file(input_path, output_path, criteria, tokenizer, min_len, max_len):
    processed = []
    criteria_format = {c: { "score": "<int>", "feedback": "<string>" } for c in criteria}
    with open(input_path, "r") as f:
        for line in tqdm(f, desc=f"Preprocessing {Path(input_path).stem}"):
            item = json.loads(line)

            trimmed = trim_transcript(item["text"], min_len, max_len)
            if not trimmed:
                continue

            item["text"] = trimmed
            input_text, output_text = build_input_output(item, criteria, criteria_format)
            
            input_ids = tokenizer(input_text, truncation=True, max_length=1536)
            label_ids = tokenizer(output_text, truncation=True, max_length=512)
            processed.append({
                "input_text": input_text,
                "output_text": output_text,
                "input_ids": input_ids["input_ids"],
                "labels": label_ids["input_ids"]
            })
            break

    with open(output_path, "w") as out:
        for item in processed:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(processed)} processed samples to {output_path}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    criteria = config["data_process"]["criteria"]
    tokenizer_name = config["preprocess"]['tokenizer']
    min_len = config["preprocess"]['min_words']
    max_len = config["preprocess"]['max_words']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Split file paths
    prefix = Path(config["data_split"]["input"]).stem
    base_dir = config["data_split"]["output_dir"]

    for split in ["train", "val", "test"]:
        in_file = f"{base_dir}/{split}/{prefix}_{split}.jsonl"
        out_file = f"{base_dir}/{split}/{prefix}_{split}_processed.jsonl"
        preprocess_file(in_file, out_file, criteria, tokenizer, min_len, max_len)