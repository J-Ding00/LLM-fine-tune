import json
import yaml
import re

def json_simple_fix(text):
    text = re.sub(r',\s*([}\]])', r'\1', text)
    if text.count('{') > text.count('}'): text += '}'
    return text

def is_valid_json_structure(obj, criteria):
    for field in criteria:
        if field not in obj:
            return False, f"Missing field: {field}"
        trait = obj[field]
        if not isinstance(trait, dict):
            return False, f"Field '{field}' is not a dict"
        if "score" not in trait or "feedback" not in trait:
            return False, f"Field '{field}' missing 'score' or 'feedback'"
        if not isinstance(trait["score"], int):
            return False, f"Score in '{field}' is not int"
        if trait["score"] < 1 or trait["score"] > 10:
            return False, f"Score in '{field}' outside range"
        if not isinstance(trait["feedback"], str):
            return False, f"Feedback in '{field}' is not str"
    return True, None

def validate_and_analyze_jsonl(path, criteria):
    with open(path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    valid = 0
    invalid_lines = []
    trait_stats = {trait: {
        "count_above_5": 0,
        "count_5_or_below": 0,
        "sum": 0,
        "scores": [],
        "total": 0
    } for trait in criteria}

    for idx, line in enumerate(lines):
        try:
            output = json.loads(line)['label']
        except Exception as e:
            invalid_lines.append((idx, f"Outer JSON error: {e}"))
            continue

        try:
            label = json.loads(output)
        except json.JSONDecodeError:
            try:
                fixed = json_simple_fix(output)
                label = json.loads(fixed)
            except Exception as e2:
                invalid_lines.append((idx, f"Unrecoverable JSON decode error: {e2}"))
                continue

        ok, msg = is_valid_json_structure(label, criteria)
        if not ok:
            invalid_lines.append((idx, msg))
            continue

        valid += 1
        for trait in criteria:
            score = label[trait]["score"]
            trait_stats[trait]["scores"].append(score)
            trait_stats[trait]["sum"] += score
            trait_stats[trait]["total"] += 1
            if score > 5:
                trait_stats[trait]["count_above_5"] += 1
            else:
                trait_stats[trait]["count_5_or_below"] += 1

    metrics = [
        f"\n{path} check:\n",
        f"Valid samples: {valid}/{total}\n",
        f"Invalid samples: {total - valid}\n",
        "\nSample errors (up to 10 shown):\n"
    ]
    for idx, err in invalid_lines[:10]:
        metrics.append(f"  Line {idx + 1}: {err}\n")

    metrics.append("\nScoring summary:\n")
    for trait, stats in trait_stats.items():
        total = stats["total"]
        if total == 0: continue
        avg = stats["sum"] / total
        metrics.append(f"\nTrait: {trait}\n")
        metrics.append(f"  Avg Score: {avg:.2f}\n")
        metrics.append(f"  >5: {stats['count_above_5']} | <=5: {stats['count_5_or_below']}\n")
        metrics.append(f"  Min: {min(stats['scores'])}, Max: {max(stats['scores'])}\n")

    return metrics

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval_file_list = config['evaluation']['jsonl_format_in']
    output_path = config['evaluation']['jsonl_format_out']
    criteria = config['webtext_load']['criteria']

    with open(output_path, "a") as out:
        for file in eval_file_list:
            metrics = validate_and_analyze_jsonl(file, criteria)
            out.writelines(metrics)
            print(f"Finished: {file}")

    print("All files processed. Metrics written to:", output_path)
