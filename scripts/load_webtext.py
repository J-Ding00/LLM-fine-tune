from datasets import load_dataset
import yaml
import json
import gc

def load_webtext_dataset(max_samples, output_file):
    """
    Load the OpenWebText dataset from Hugging Face.

    Args:
        split (str): Dataset split to load (e.g., "train").
        max_samples (int or None): If set, limit to the first N samples.

    Returns:
        List[str]: List of text documents.
    """

    dataset = load_dataset("stas/openwebtext-10k", split='train', streaming=True)
    with open(output_file, "w") as f:
        iterator = iter(dataset)
        for i in range(max_samples):
            item = next(iterator)
            item['id'] = i+1
            f.write(json.dumps(item) + "\n")
    del(iterator)
    gc.collect()

if __name__ == "__main__":
    config_path = 'config.yaml'

    # Load configuration from the YAML file.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_samples = config['webtext_load']['num_samples']
    output_file = config['webtext_load']['raw_output_file']

    load_webtext_dataset(num_samples, output_file)
