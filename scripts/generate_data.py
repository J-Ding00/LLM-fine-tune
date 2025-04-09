# Only use when data is very limited
import yaml
import json
from clients import openai_client

def generate_transcript(criteria, num_per_criteria_category, temperature, max_tokens, gpt_model):
    """
    Generates a synthetic speech transcript using the given GPT model based on a provided prompt.
    """

    good_prompt = f"""
    Write me a paragraph with some {criteria}, only output the paragraph itself.
    """

    bad_prompt = f"""
    Write me a paragraph that lacks of {criteria}, only output the paragraph itself.
    """
    transcripts = []
    for _ in range(num_per_criteria_category):
        response = openai_client.responses.create(
            model=gpt_model,
            input=good_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        transcripts.append(response.output_text.strip())

        response = openai_client.responses.create(
            model=gpt_model,
            input=bad_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        transcripts.append(response.output_text.strip())
    return transcripts

if __name__ == "__main__":
    config_path = 'config.yaml'
    
    # Load configuration from the YAML file.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_samples = config['raw_transcripts_gen']['num_samples_per_criteria']
    output_file = config['raw_transcripts_gen']['output_file']
    criteria = config['raw_transcripts_gen']['criteria']
    temperature = config['raw_transcripts_gen']['temperature']
    max_tokens = config['raw_transcripts_gen']['max_tokens']
    gpt_model = config['openai']['generate_model']

    idx = 1
    for c in criteria:
        transcripts = []
        print(f"Generating {num_samples*2} transcripts in criteria {c} ...")
        try:
            transcript = generate_transcript(c, num_samples, temperature, max_tokens, gpt_model)
            for t in transcript:
                transcripts.append({
                    "id": idx,
                    "criteria": c,
                    "transcript": t
                })
                idx += 1

            # Save the transcripts as JSON Lines (one JSON object per line).
            with open(output_file, "a") as f:
                for item in transcripts:
                    f.write(json.dumps(item) + "\n")

        except Exception as e:
            print(f"Error generating transcripts in criteria {c}")

    