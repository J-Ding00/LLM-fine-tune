from clients import openai_client
import json
import time
import yaml

def generate_feedback(text, criteria, criteria_format, model, max_token_limit, max_retries=2):
    instruction = f"""
    You are an expert speech evaluator. Given a transcript of a speaker's message, assess the communication quality in terms of the following traits:
    {criteria}

    For each trait, assign a score from 1 to 10, where 10 is excellent and 1 is very poor. Then provide a short explanation (1–2 sentences) justifying your score for that trait.

    The final output must be a JSON string with the exact format below:
    {criteria_format}
    """

    transcript = f"""
    Transcript:
    {text}
    """

    for attempt in range(max_retries):
        try:
            response = openai_client.responses.create(
                model=model,
                instructions=instruction,
                input=transcript,
                max_output_tokens=max_token_limit,
                temperature=0.2
            )
            return response.output_text.strip().removeprefix('```json\n').removesuffix('\n```')
        except Exception as e:
            print(f"[Warning] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise RuntimeError("labeling failed after retries")

def label_batch(input_path, output_path, criteria, model, max_token_limit):
    with open(input_path, 'r') as f:
        lines = [json.loads(l) for l in f]
    criteria_format = {c: { "score": "<int>", "feedback": "<string>" } for c in criteria}
    with open(output_path, 'w') as fout:
        for example in lines:
            transcript = example["text"]
            id = example["id"]
            try:
                feedback = generate_feedback(transcript, criteria, criteria_format, model, max_token_limit)
                labeled = {"id":id, "text": transcript, "label": feedback}
                fout.write(json.dumps(labeled) + "\n")
            except Exception as e:
                print(f"[Error] Skipping sample {id} due to failure: {e}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_file = config['webtext_load']['raw_output_file']
    output_file = config['webtext_load']['label_output_file']
    criteria = config['webtext_load']['criteria']
    max_token_limit = config['webtext_load']['max_eval_tokens']
    model = config['openai']['generate_model']

    label_batch(input_file, output_file, criteria, model, max_token_limit)