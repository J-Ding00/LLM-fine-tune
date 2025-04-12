import os
import json

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
    average_transcript_length('data/raw/openwebtext_sample.jsonl')
    exit()