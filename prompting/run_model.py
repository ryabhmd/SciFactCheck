import os
import re
import json
import argparse
from datasets import load_dataset
from model_interface import HFModel, LiteLLMModel, vLLMModel
from dotenv import load_dotenv
from tqdm import tqdm 
from string import Template

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", choices=["hf", "litellm", "vllm"], required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel partitions in case on vLLM.")
    parser.add_argument("--dataset_name", required=True, help="HuggingFace dataset identifier, e.g. 'username/dataset_name'")
    parser.add_argument("--prompt_template", required=True, help="Path to .txt file with prompt template (use {scientific_entity} and {field} placeholders)")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--hf_api_key", default=os.getenv("HF_API_KEY"))
    parser.add_argument("--litellm_api_key", default=os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="Whether to format prompts with tokenizer.apply_chat_template() (default: False)")
    parser.add_argument("--two_prompts", action="store_true", default=False, help="Whether to adapt a strategy of two follow-up prompts instead of only one.")
    parser.add_argument("--citation_prompt_template", required=False, help="Path to .txt file with the citations prompt template in case a two-prompt strategy is adopted.")
    parser.add_argument("--from_checkpoint", action="store_true", default=False, help="Whether to continue prompting from a specific checkpoint of the dataset instead of from scratch. If chosen, the argument --checkpoint_path should be passed.")
    parser.add_argument("--checkpoint_path", help="If continuing prompting from a checkpoint, denote the path to load the data from.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------Load prompt template------------
    with open(args.prompt_template, "r") as f:
        template_text = f.read().strip()
    template = Template(template_text)

    if args.two_prompts:
        with open(args.citation_prompt_template, "r") as f:
            citation_template_text = f.read().strip()
        citation_template = Template(citation_template_text)

    # -----------Load dataset from HF Hub------------
    print(f"Loading dataset {args.dataset_name} from HuggingFace Hub...")
    dataset = load_dataset(args.dataset_name, token=args.hf_api_key)
    if args.limit:
        dataset = dataset.select(range(args.limit))
    print(f"Loaded {len(dataset)} samples.")

    # ----------- Initialize model ------------
    if args.model_type == "hf":
        model = HFModel(
            args.model_name,
            api_key=args.hf_api_key,
            force_download=True,
            apply_chat_template=args.apply_chat_template,
        )
        
    elif args.model_type == "litellm":
        model = LiteLLMModel(
            args.model_name,
            api_key=args.litellm_api_key,
            )
        
    elif args.model_type == "vllm":
        model = vLLMModel(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            is_chat=args.apply_chat_template,
            return_logprobs=True,           
            )     

    # ----------Output setup-------------
    # If checkpoint is given, continue from last element
    if args.from_checkpoint:
        print(f"Loading data from checkpoint {args.checkpoint_path}")
        input_data = []
        with open(args.checkpoint_path, "r") as f:
            for line in f:
                if line.strip():
                    input_data.append(json.loads(line))
        # create output path
        checkpoint_index = input_data[-1]["id"]
        output_path = os.path.join(
            args.output_dir, f"{args.model_name.replace('/', '_')}_results_checkpoint_{checkpoint_index}.jsonl"
        )
        data = dataset['train'].select(range(checkpoint_index+1, len(dataset['train'])))
        
    # Otherwise, start from scratch    
    else:
        output_path = os.path.join(
            args.output_dir, f"{args.model_name.replace('/', '_')}_results.jsonl"
        )
         # in case no checkpoint is given, to start prompting loop from 0
        checkpoint_index = -1
        data = dataset['train']
        
    fout = open(output_path, "w")

    # ----------Run generation loop-------------
    for i, row in enumerate(tqdm(data, desc="Prompting entities"), start=checkpoint_index+1):
        entity = row.get("scientific_entity")
        field = row.get("field")

        prompt = template.substitute(scientific_entity=entity, field=field)

        result = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        
        if args.two_prompts:
            paragraph_result = result["text"]
            # remove <think> boxes in the paragraph so they are not included in the citation prompt
            if "<think>" in paragraph_result:
                paragraph_result = re.sub(r"<think>.*?</think>", "", paragraph_result, flags=re.DOTALL)
                
            citation_prompt = citation_template.substitute(paragraph=paragraph_result)
            citation_result = model.generate(citation_prompt, max_new_tokens=args.max_new_tokens)

            record = {
            "id": i,
            "field": field,
            "entity": entity,
            "two_prompts": args.two_prompts,
            "output": result,
            "output_citation": citation_result
        }
            
        else:
            record = {
            "id": i,
            "field": field,
            "entity": entity,
            "two_prompts": args.two_prompts,
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

