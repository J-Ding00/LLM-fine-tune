import json
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def build_prompt(text, criteria, criteria_format):
    instruction = f"""
    You are an expert speech evaluator. Given a transcript of a speaker's message, assess the communication quality in terms of the following traits:
    {criteria}

    For each trait, assign a score from 1 (poor) to 10 (excellent). Be honest and critical in your assessment without bias toward high scores. 
    Then provide a short explanation (1–2 sentences) justifying your score for that trait.

    The final output must be a JSON string with the exact format below:
    {criteria_format}
    """.strip()

    transcript = f"Transcript:\n{text}"
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{transcript}"}
    ]
    return messages

def generate_local_feedback(model, tokenizer, text, criteria, criteria_format, max_new_tokens):
    messages = build_prompt(text, criteria, criteria_format)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2
    )

    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Clean JSON artifacts if wrapped
    # if decoded.startswith("```json"):
    #     decoded = decoded.removeprefix("```json").removesuffix("```").strip()
    return decoded

def label_batch_local(input_path, output_path, criteria, model, tokenizer, max_new_tokens):
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    criteria_format = {c: { "score": "<int>", "feedback": "<string>" } for c in criteria}

    with open(output_path, "w") as fout:
        for example in tqdm(lines, desc="Labeling samples"):
            transcript = example["text"]
            try:
                feedback = generate_local_feedback(model, tokenizer, transcript, criteria, criteria_format, max_new_tokens)
                labeled = {"text": transcript, "label": feedback}
                fout.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Error] Skipping sample")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_file = config['pretrain_model']['eval']['input_test_path']
    output_file = config['pretrain_model']['eval']['pretrain_output']
    criteria = config["data_process"]["criteria"]
    max_new_tokens = config['pretrain_model']['eval']["max_eval_tokens"]
    model_name = config["pretrain_model"]["model"]

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    label_batch_local(input_file, output_file, criteria, model, tokenizer, max_new_tokens)