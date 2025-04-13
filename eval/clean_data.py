import json
import re
import yaml

def json_simple_fix(text):
    text = re.sub(r',\s*([}\]])', r'\1', text)
    if text.count('{') > text.count('}'): text += '}'
    return text

def is_valid_json_structure(obj, criteria):
    for field in criteria:
        if field not in obj:
            return False
        trait = obj[field]
        if not isinstance(trait, dict):
            return False
        if "score" not in trait or "feedback" not in trait:
            return False
        if not isinstance(trait["score"], int):
            return False
        if trait["score"] < 1 or trait["score"] > 10:
            return False
        if not isinstance(trait["feedback"], str):
            return False
    return True

def filter_valid_examples(path, criteria, output_path):
    valid_lines = 0
    with open(path, "r") as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(fin):
            try:
                outer = json.loads(line)
                raw_label = outer.get("label")
                try:
                    label = json.loads(raw_label)
                except json.JSONDecodeError:
                    label = json.loads(json_simple_fix(raw_label))

                if is_valid_json_structure(label, criteria):
                    valid_lines += 1
                    fout.write(line)
            except Exception:
                continue

    print(f"Filtered and saved {valid_lines} valid examples to {output_path}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # eval_file_list = config['evaluation']['jsonl_format_in']
    # output_file = config['evaluation']['clean_webtext_data_out']

    eval_file_list = config['evaluation']['jsonl_format_in']
    output_file = config['evaluation']['clean_generated_data_out']

    criteria = config['data_process']['criteria']

    for file in eval_file_list:
        filter_valid_examples(file, criteria, output_file)
