"""
High-level wrapper for running multiple evaluation methods on model outputs.

This module defines the `EvaluationRunner` class, which orchestrates
different evaluations defined in `model_evaluator.py` and applies them to every record in a model output dataset.

- Current list of eval options indications: 
--- ovr-logprobs
--- ovr-uncertainty
--- attr 

Functionalities:
- Load model outputs from a JSONL file.
- Run selected evaluation methods
- Save the enriched dataset with per-sample evaluation results.
- Optionally generate quick aggregate summaries (per field or model).

Notes:
- Use `run_eval.py` for command-line execution.
"""

import json
from tqdm import tqdm
from model_evaluator import ModelEvaluator

class EvaluationRunner:
    """
    Orchestrates evaluations on a dataset of model outputs.

    This class acts as the bridge between model outputs (JSONL files)
    and the evaluation logic in ModelEvaluator.
    """
     
    def __init__(self, model_name, file_path):
        self.model_name = model_name
        self.file_path = file_path
        self.data = self._load_jsonl(file_path)
        self.evaluator = ModelEvaluator(model_name, data=self.data)
        print(f"Loaded {len(self.data)} records for model {self.model_name}")

    @staticmethod
    def _load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def run(self, eval_types):
        """
        Run one or more evaluation types over the dataset.

        Args:
            eval_types (list[str]): Evaluation methods to run.
                Options: ["ovr-logprobs", "ovr-uncertainty", "attr"]

        This function enriches each record with the corresponding evaluation results, stored under keys like "eval_uncertainty" or "eval_attr".
        """

        print(f"Running evaluations: {eval_types} on output of {self.model_name}")

        for record in tqdm(self.data, desc=f"Evaluating {self.model_name}"):
            for eval_type in eval_types:

                if eval_type == "ovr-logprobs":
                    record["output"]["logprob_eval"] = record["output"].get("length_norm_logprob")

                elif eval_type == "ovr-uncertainty":
                    text = record["output"]["text"]
                    mean_cert, aspect_scores = self.evaluator.get_uncertainty_estimations(text)
                    record["eval_uncertainty"] = {
                        "mean_sentence_certainty": mean_cert,
                        "aspect_scores": aspect_scores,
                    }

                elif eval_type == "attr":
                    text = record["output"]["text"]
                    score = self.evaluator.process_attribution_hallucinations(text)
                    record["eval_attr"] = {"attr_score": score}

        print("Done evaluating samples!")

    def save(self, output_path):
        with open(output_path, "w") as f:
            for record in self.data:
                f.write(json.dumps(record) + "\n")
        print(f"Saved enriched data to {output_path}")

    ## TODO: create functions that aggregate results of evals by field and compare between them

    def summarize(self):
        # Aggregate summaries (logprob, uncertainty, etc.)
        print("Summary per field:")
        self.evaluator.eval_logprobs()  # reuses the existing aggregation
        self.evaluator.compare_fields()

