# Simple script to combine results for the same model in different checkpoints
import json
import argparse 
from pathlib import Path

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--main_data_path", required=True)
  parser.add_argument("--checkpoint_data_path", required=True)
  
  args = parser.parse_args()
  
  data_path = Path(args.main_data_path)
  checkpoint_path = Path(args.checkpoint_data_path)

  combined_path = data_path.stem + "_combined_checkpoints.jsonl"
  
  with open(data_path, 'r') as f:
    data = f.readlines()
    data = [json.loads(claim) for claim in data]

  with open(checkpoint_path, 'r') as f:
    data_checkpoint = f.readlines()
    data_checkpoint = [json.loads(claim) for claim in data_checkpoint]

  assert len(data) + len(data_checkpoint) == 2500

  data.extend(data_checkpoint)

  with open(combined_path, 'w') as f:
    for item in data:
      f.write(json.dumps(item) + '\n')

  print(f"Saved combined checkpoints at: {combined_path}")


if __name__ == "__main__":
    main()
