import json

REQUIRED_FIELDS = ['formality', 'persuasiveness', 'enthusiasm', 'empathy', 'filler words', 'transition logic']

def is_valid_json_structure(obj):
    for field in REQUIRED_FIELDS:
        if field not in obj:
            return False, f"Missing field: {field}"
        trait = obj[field]
        if not isinstance(trait, dict):
            return False, f"Field '{field}' is not a dict"
        if "score" not in trait or "feedback" not in trait:
            return False, f"Field '{field}' missing 'score' or 'feedback'"
        if not isinstance(trait["score"], int):
            return False, f"Score in '{field}' is not int"
        if not isinstance(trait["feedback"], str):
            return False, f"Feedback in '{field}' is not str"
    return True, None

def evaluate_jsonl(path):
    with open(path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    valid = 0
    invalid_lines = []

    for idx, line in enumerate(lines):
        output = json.loads(line)['label']
        try:
            label = json.loads(output)
            ok, msg = is_valid_json_structure(label)
            if ok:
                valid += 1
            else:
                invalid_lines.append((idx, msg))
        except Exception as e:
            invalid_lines.append((idx, f"JSON decode error: {e}"))

    print(f"Valid samples: {valid}/{total}")
    if invalid_lines:
        print("Invalid examples:")
        for idx, err in invalid_lines[:10]:  # only show first 10
            print(f"  Line {idx + 1}: {err}")
    return valid, total, invalid_lines

evaluate_jsonl('data/labeled/label_openwebtext_sample.jsonl')