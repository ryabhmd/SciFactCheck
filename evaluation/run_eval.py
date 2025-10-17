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
    parser.add_argument("--eval", nargs="+", required=True)
    args = parser.parse_args()

    input_path = Path(args.file)

    evaluated_path = input_path.with_name(input_path.stem + "_evaluated" + input_path.suffix)
    summary_path = input_path.with_name(input_path.stem + "_summary" + input_path.suffix)

    runner = EvaluationRunner(args.model_name, args.file)
    runner.run(args.eval)
    runner.save(evaluated_path)
    runner.save_summary_evaluations(args.eval, summary_path)
