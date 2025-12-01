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
from collections import defaultdict
from statistics import mean, stdev
from prep_file_veriscore import clean_model_output

class EvaluationRunner:
    """
    Orchestrates evaluations on a dataset of model outputs.

    This class acts as the bridge between model outputs (JSONL files)
    and the evaluation logic in ModelEvaluator.
    """
     
    def __init__(self, model_name, file_path, output_path):
        self.model_name = model_name
        self.file_path = file_path
        self.data = self._load_jsonl(file_path)
        self.evaluator = ModelEvaluator(model_name, data=self.data)
        self.output_path = output_path
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
                    record["eval_logprobs"] = record["output"]["length_norm_logprob"]

                if eval_type == "ovr-uncertainty":
                    text = record["output"]["text"]
                    text = clean_model_output(text)
                    mean_cert, aspect_scores = self.evaluator.get_uncertainty_estimations(text)
                    if mean_cert != None:
                        mean_cert = float(mean_cert)
                    record["eval_uncertainty"] = {
                        "mean_sentence_certainty": mean_cert,
                        "aspect_scores": aspect_scores,
                    }

                elif eval_type == "attr":
                    text = record["output"]["text"]
                    score = self.evaluator.process_attribution_hallucinations(text)
                    print(score)
                    print(type(score))
                    record["eval_attr"] = {"attr_score": score}

            self.save()

        print("Done evaluating samples!")

    # =========================== Collecting and saving results =========================== #

    def collect_attr_by_field(self):
        """
        Collects results of attribution hallucination percentages and
        1. Prints summary of the mean percentage per field.
        2. Returns a dictionary containing the summary per field with the mean score, stdev, and number of collected samples.
        """
        
        field_attr = defaultdict(list)
        
        for record in tqdm(self.data, desc="Processing outputs"):
            field = record["field"]
            attr_score = record["eval_attr"]["attr_score"]
            if field and attr_score is not None:
                field_attr[field].append(attr_score)
                
        # Print summary
        print("=== attribution hallucination summary by field ===")
        field_attr_summary = defaultdict(dict)
        for field, values in field_attr.items():
            print(f"{field:20s} | {len(values):3d} samples | mean = {mean(values):.8f}")
            field_attr_summary[field] = {
                "mean": mean(values),
                "stdev": stdev(values),
                "samples": len(values)
                }
        
        return dict(field_attr_summary)
    
    def collect_logprobs_by_field(self):
        """
        Collects results of logprobs and
        1. Prints summary of the mean length-normalized logprob scores per field.
        2. Returns a dictionary containing the summary per field with the mean score, stdev, and number of collected samples.
        """
        field_logprobs = defaultdict(list)
        
        for record in tqdm(self.data, desc="Processing outputs"):
            field = record["field"]
            logprob = record["eval_logprobs"]
            if field and logprob is not None:
                field_logprobs[field].append(logprob)
                
        # Print summary
        print("=== logprob summary by field ===")
        field_logprob_summary = defaultdict(dict)
        for field, values in field_logprobs.items():
            print(f"{field:20s} | {len(values):3d} samples | mean = {mean(values):.8f}")
            field_logprob_summary[field] = {
                "mean": mean(values),
                "stdev": stdev(values),
                "samples": len(values)
                }
            
        return dict(field_logprob_summary)
    

    def collect_uncertainty_by_field(self):
       """
        Collects results of uncertainty evaluations and
        1. Prints summary of the mean uncertainty scores (sentence- and aspect-level) per field.
        2. Returns two dictionaries (sentence-level and aspect-level) containing the summary per field with the mean score, stdev, and number of collected samples.
        """
       
       field_sentence_uncertainty = defaultdict(list)
       field_aspect_uncertainty = defaultdict(list)
       
       for record in tqdm(self.data, desc="Processing outputs"):
          field = record["field"]
          sentence_uncertainty = record["eval_uncertainty"]["mean_sentence_certainty"]
          if field and sentence_uncertainty is not None:
             field_sentence_uncertainty[field].append(sentence_uncertainty)
             
          aspect_uncertainty = record["eval_uncertainty"]["aspect_scores"]
          if field and aspect_uncertainty is not None:
             field_aspect_uncertainty[field].append(aspect_uncertainty)
             
       print("=== mean sentence certainty summary by field ===")
       field_sentence_summary = defaultdict(dict)
       for field, values in field_sentence_uncertainty.items():
          print(f"{field:20s} | {len(values):3d} samples | mean = {mean(values):.8f}")
          field_sentence_summary[field] = {
              "mean": mean(values),
              "stdev": stdev(values),
              "samples": len(values)
              }
          
       print("=== mean aspect certainty summary by field ===") # for each field, go over the aspect_level_uncertainty per each aspect and get the mean
       field_aspect_summary = defaultdict(dict)
       for field, values in field_aspect_uncertainty.items():
           field_specific_dict = defaultdict(list)
           for value in values:
               for aspect, score in value.items():
                   if score is not None:
                       field_specific_dict[aspect].append(score)
                       
           for aspect, values in field_specific_dict.items():
               print(f"{field:20s} | {aspect:20s} | {len(values):3d} samples | mean = {mean(values):.8f}")
               field_aspect_summary[field][aspect] = {
                   "mean": mean(values),
                   "stdev": stdev(values),
                   "samples": len(values)
                   }
               
       return dict(field_sentence_summary), dict(field_aspect_summary)

    def save(self):
        with open(self.output_path, "w") as f:
            for record in self.data:
                f.write(json.dumps(record) + "\n")
        print(f"Saved evaluated data to {self.output_path}")
    
    def save_summary_evaluations(self, eval_types, output_path):
        """
        Goes over data and according to the given eval_types, collects summaries per field (mean value, stdev, number of samples) and saves a JSON including all summaries per eval type. 
        """

        eval_types_summaries = defaultdict(dict)
        for eval_type in eval_types:
            if eval_type == "attr":
                attr_sum = self.collect_attr_by_field()
                eval_types_summaries["attr"] = attr_sum
                
            elif eval_type == "ovr-logprobs":
                logprobs_sum = self.collect_logprobs_by_field()
                eval_types_summaries["logprobs"] = logprobs_sum
                
            elif eval_type == "ovr-uncertainty":
                sentence_sum, aspect_sum = self.collect_uncertainty_by_field()
                eval_types_summaries["sentence_uncertainty"] = sentence_sum
                eval_types_summaries["aspect_uncertainty"] = aspect_sum
                
        
        with open(output_path, "w") as f:
            json.dump(eval_types_summaries, f)
            
        print(f"Saved evaluation summaries to {output_path}.")

