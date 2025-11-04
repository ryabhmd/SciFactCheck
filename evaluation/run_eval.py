"""
Command-line interface for running one or more evaluation methods on LLM-generated outputs.

This script acts as a front-end for the `EvaluationRunner` class.
It allows you to specify which evaluations to run, on which model results, and where to save the enriched output.

Arguments:
    --model_name   Name or identifier of the evaluated model
    --file         Path to JSONL file with model outputs
    --save         Path to save the enriched JSONL file
    --eval         List of evaluations to run (choices: ovr-logprobs, ovr-uncertainty, attr,)

Outputs:
    A new JSONL file where each record contains:
        - The original model output
        - One or more evaluation results (stored under "eval_*" keys)

"""

import argparse
from pathlib import Path
from evaluation_runner import EvaluationRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--eval", nargs="+", required=True)
    args = parser.parse_args()

    input_path = Path(args.file)
    save_path = Path(args.save)

    evaluated_filename = input_path.stem + "evaluated.json"
    evaluated_path = save_path / evaluated_filename
    summary_filename = input_path.stem + "_summary.json"
    summary_path = save_path / summary_filename

    print(f"Evaluated file path: {evaluated_path}")
    print(f"Summary file path: {summary_path}")

    runner = EvaluationRunner(args.model_name, args.file, output_path=evaluated_path)
    runner.run(args.eval)
    runner.save()
    runner.save_summary_evaluations(args.eval, summary_path)
