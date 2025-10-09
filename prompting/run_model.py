import os
import json
import argparse
from datasets import load_dataset
from model_interface import HFModel, LiteLLMModel
from dotenv import load_dotenv
from tqdm import tqdm 

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", choices=["hf", "litellm"], required=True)
    parser.add_argument("--dataset_name", required=True, help="HuggingFace dataset identifier, e.g. 'username/dataset_name'")
    parser.add_argument("--prompt_template", required=True, help="Path to .txt file with prompt template (use {scientific_entity} and {field} placeholders)")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--hf_api_key", default=os.getenv("HF_API_KEY"))
    parser.add_argument("--litellm_api_key", default=os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------Load prompt template------------
    with open(args.prompt_template, "r") as f:
        template = f.read().strip()

    # -----------Load dataset from HF Hub------------
    print(f"Loading dataset {args.dataset_name} from HuggingFace Hub...")
    dataset = load_dataset(args.dataset_name, token=args.hf_api_key)
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"Loaded {len(dataset)} samples.")

    # -----------Initialize model------------
    if args.model_type == "hf":
        model = HFModel(args.model_name, api_key=args.hf_api_key, force_download=True)
    else:
        model = LiteLLMModel(args.model_name, api_key=args.litellm_api_key)

    # ----------Output setup-------------
    output_path = os.path.join(
        args.output_dir, f"{args.model_name.replace('/', '_')}_results.jsonl"
    )
    fout = open(output_path, "w")

    # ----------Run generation loop-------------
    for i, row in enumerate(tqdm(dataset['train'], desc="Prompting entities")):
        entity = row.get("scientific_entity")
        field = row.get("field")

        prompt = template.format(scientific_entity=entity, field=field)

        result = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        record = {
            "id": i,
            "entity": entity,
            "prompt": prompt,
            "output": result,
        }

        fout.write(json.dumps(record) + "\n")
        fout.flush()

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} entities...")

    fout.close()
    print(f"Finished. Results saved to: {output_path}")


if __name__ == "__main__":
    main()

