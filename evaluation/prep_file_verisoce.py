import argparse
from tqdm import tqdm
from pathlib import Path

import re
import os
import json

def _load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

def create_verisocre_input_files(data, output_path):
    """
    Create Veriscore input files for each record in a given data.
    
    For each record in `data`, this function:
    - Removes the citations section from the output text
    - Extracts 'prompt', 'prompt_source', 'response', and 'model'
    - Saves a JSONL file per record under:
        output_path/<model_name_sanitized>/<index>_<model_name_sanitized>.jsonl
    """

    for idx, record in enumerate(tqdm(data, desc="Creating Veriscore input files")):

        text = record["output"]["text"]

        pattern = r'\{\s*"citations"\s*:\s*\[.*?\]\s*\}'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            text = text[:match.start()]

        prompt = record["prompt"]
        prompt_source = "template"

        response = text

        model = record["output"]["meta"]["model"]

        # Sanitize model name (remove slashes and backslashes)
        model_folder = re.sub(r'[\\/]', '', model)

        # Create model-specific folder
        model_dir = Path(output_path) / model_folder
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build file name with padded index
        filename = f"{idx:03d}_{model_folder}.jsonl"
        file_path = model_dir / filename

        # Build record to save
        veriscore_record = {
            "prompt": prompt,
            "prompt_source": prompt_source,
            "response": response,
            "model": model
        }

        # Write JSONL file (1 record per file)
        with open(file_path, "w", encoding="utf-8") as fout:
            json.dump(veriscore_record, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    data = _load_jsonl(args.data_path)

    create_verisocre_input_files(data, args.save_dir)
