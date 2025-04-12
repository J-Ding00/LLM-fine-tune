# Only use when data is very limited
import yaml
import json
from clients import openai_client
import random

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

def generate_negative_transcript_samples(n, trait, model, temperature, max_out_token, real_samples, output_file):
    """
    Generate `n` long-form samples recursively from GPT-4o-mini with chat history.
    """

    system_prompt = f"""
    You are a helpful assistant generating realistic speech transcript samples for training a speech evaluation model.

    The transcript should intentionally and subtly lack the trait of "{trait}". Use natural, unscripted, and conversational language that reflects the absence of this trait. Do not mention the trait explicitly.

    Each transcript should follow similar style, tone, and structure as the provided example — but use completely different content. Do not copy or closely paraphrase the original. The output should be similar to the provided example in length.

    Each transcript should be distinct in content, tone, and structure from your previously generated transcripts.

    Avoid including the title, and common speech openings like 'Hi,', 'So,', 'Hey,' or 'You know,' etc. Only output the content of transcript text.
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]

    with open(output_file, "a") as f:
        for i in range(1, n + 1):

            try:
                user_prompt = f"""
                {random.choice(real_samples)}
                """.strip()

                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages+[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    max_tokens=max_out_token  # adjust as needed for target length
                )

                assistant_reply = response.choices[0].message.content.strip()

                # Add to chat history
                messages.append({"role": "assistant", "content": assistant_reply})
                f.write(json.dumps({'id':i, 'trait':trait, 'text':assistant_reply}) + "\n")


            except Exception as e:
                print(f"⚠️ Error at sample {i}: {e}")
                break

if __name__ == "__main__":
    config_path = 'config.yaml'
    
    # Load configuration from the YAML file.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_samples = config['raw_transcripts_gen']['num_samples']
    output_file = config['raw_transcripts_gen']['output_file']
    criteria = config['raw_transcripts_gen']['minority_criteria']
    temperature = config['raw_transcripts_gen']['temperature']
    max_tokens = config['raw_transcripts_gen']['max_tokens']
    gpt_model = config['openai']['generate_model']
    real_sample_path = config['webtext_load']['raw_output_file']

    with open(real_sample_path, "r") as f:
        real_samples = [json.loads(l)['text'].strip() for l in f]

    for trait in criteria:
        generate_negative_transcript_samples(num_samples, trait, gpt_model, temperature, max_tokens, real_samples, output_file)

    