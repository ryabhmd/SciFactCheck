import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

def _load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_veriscore_input_file(data, output_path):
    """
    Create a single Veriscore input JSONL file for all records.
    
    Each line is a dictionary with keys:
    - prompt
    - prompt_source
    - response
    - model
    """

    if not data:
        print("No data found!")
        return

    # Assume all records belong to the same model
    model = data[0]["output"]["meta"]["model"]
    model_folder = re.sub(r'[\\/]', '', model)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file: <model_name>_veriscore_input.jsonl
    output_file = output_dir / f"{model_folder}_veriscore_input.jsonl"

    with open(output_file, "w", encoding="utf-8") as fout:
        for record in tqdm(data, desc=f"Preparing Veriscore input for {model_folder}"):
            text = record["output"]["text"]

            # Remove citations section if present
            pattern = r'\{\s*"citations"\s*:\s*\[.*?\]\s*\}'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = text[:match.start()].strip()

            prompt = record["prompt"]
            prompt_source = "template"
            response = text

            veriscore_record = {
                "prompt": prompt,
                "prompt_source": prompt_source,
                "response": response,
                "model": model
            }

            fout.write(json.dumps(veriscore_record, ensure_ascii=False) + "\n")

    print(f"Veriscore input saved at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    data = _load_jsonl(args.data_path)
    create_veriscore_input_file(data, args.save_dir)

