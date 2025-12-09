"""
Core evaluation class for analyzing LLM-generated scientific outputs.

This module defines the `ModelEvaluator` class, which provides low-level
functions to analyze a model's outputs at both sample and aggregate levels.
It directly operates on nested dictionary data.

Main functionalities based on factuality hallucination type:

1. OVR 
-- Logprob evaluation: compute and summarize model confidence by field.
-- Linguistic uncertainty estimation: quantify certainty in generated text using a pretrained `CertaintyEstimator` model.

2. ATTR: check if cited papers are valid.

- Utility methods for reading, saving, and aggregating results.

Notes:
- This file focuses on the evaluation *logic*, not orchestration.
- For running and combining multiple evaluations, see `evaluation_runner.py`.
"""


import json
import os
import requests
import re
import time
import string
import unicodedata
from rapidfuzz import fuzz, process

import pandas as pd
from collections import defaultdict
from statistics import mean, stdev
from dotenv import load_dotenv
from tqdm import tqdm

from certainty_estimator.predict_certainty import CertaintyEstimator

class ModelEvaluator:
    """
    Handles loading model outputs and running evaluation methods directly on nested dict data.
    """

    def __init__(self, model_name: str, data=None, file_path=None):
        """
        Initialize evaluator for a given model.
        Provide either a list of dicts (`data`) or a JSONL `file_path`.
        """
        self.model_name = model_name

        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = self._read_jsonl(file_path)
        else:
            raise ValueError("Provide either `data` or `file_path`.")

        print(f"Loaded {len(self.data)} records for model {self.model_name}")
        self.results = {}

    @staticmethod
    def _read_jsonl(path):
        """Helper to read JSONL file into a list of dicts."""
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    # -------------------- Evaluation methods: OVR --------------------

    # ---- logprobs ----
    def eval_logprobs(self):
        """
        Collect length-normalized logprobs grouped by field.
        """
        field_logprobs = defaultdict(list)

        for record in self.data:
            field = record["field"]
            logprob = record["output"]["length_norm_logprob"]
            if field and logprob is not None:
                field_logprobs[field].append(logprob)

        summary = []
        for field, values in field_logprobs.items():
            summary.append({
                "field": field,
                "count": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            })

        self.results["logprob_summary"] = summary
        print("Logprob summary:")
        for s in summary:
            print(f"{s['field']:35s} | {s['count']:4d} samples | mean = {s['mean']:.6f}")
        return summary

    def compare_fields(self):
        """Rank fields by mean logprob (confidence)."""
        if "logprob_summary" not in self.results:
            self.eval_logprobs()
        ranked = sorted(
            self.results["logprob_summary"], key=lambda x: x["mean"], reverse=True
        )
        print("Field ranking by confidence:")
        for i, r in enumerate(ranked, start=1):
            print(f"{i:2d}. {r['field']:35s} | mean = {r['mean']:.6f}")
        return ranked
    
    # ---- linguistic uncertainty ---- ** Running CertaintyEstimator works best with transformers==4.40.2 ** 
    def aspect_level_uncertainty_score(self, aspect_dict):
        """
        Score per aspect type in percentage: high = certain.
        The percentage means how many sentences in the LLM output are predicted as certain for the aspect.
        """
        
        aspects = ['Probability', 'Number', 'Extent', 'Condition', 'Framing', 'Suggestion']
        aspect_scores = {}
        
        for aspect in aspects:
            if aspect_dict[aspect]:
                results = 0
                for certainty in aspect_dict[aspect]:
                    if certainty == 'Certain':
                        results += 1
                aspect_scores[aspect] = results/len(aspect_dict[aspect])
            else:
                aspect_scores[aspect] = None
        
        return aspect_scores

    def get_uncertainty_estimations(self, text: str):
        """
        Given a text (output from LLM), the function:
        1. Removes citations from text 
        2. Strips it to sentences
        3. Predicts the sentence-level certainty of each sentence (scale of 1-6)
        4. Predicts the aspect-level certainty of each sentence (binary: certain/uncertain over six possible aspects: number, extent, probability, condition, framing, and suggestion)

        Returns
        1. mean of sentence-level certainty of the text (scale of 1-6, higher is more certain)
        2. percentage of certain sentences for each aspect (scale of 0-1, higher is more certain)
        """
        
        pattern = r'\{\s*"citations"\s*:\s*\[.*?\]\s*\}'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            text = text[:match.start()]
            
        sentences = text.split('.')
        sentences = [sentence.strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences if sentence != '']

        if sentences != []:
        
            aspect_dict = defaultdict(list)
            
            # sentence-level certainty prediction
            sentence_level_estimator = CertaintyEstimator('sentence-level')
            sentence_level_certainty = sentence_level_estimator.predict(sentences,tqdm=tqdm)
            mean_sentence_certainty = mean(sentence_level_certainty)
            
            # aspect-level certainty prediction
            aspect_level_estimator = CertaintyEstimator('aspect-level')
            aspect_level_certainty = aspect_level_estimator.predict(sentences,tqdm=tqdm)
            
            for sentence_result in aspect_level_certainty:
                for result in sentence_result:
                    aspect, certainty = result[0], result[1]
                    aspect_dict[aspect].append(certainty)
                    
            aspect_scores = self.aspect_level_uncertainty_score(aspect_dict)

        else:
            return None, None
        
        return mean_sentence_certainty, aspect_scores
    

    def collect_certainty_by_field(self, data):
        """
        Collects uncertainty results (both sentence-level and aspect-level) per sample in data. 
        Returns two dictionaries:
        1. sentence-level certainty results per field
        2. aspect-level certainty results per field
        """
        
        field_sentence_certainty = defaultdict(list)
        field_aspect_certainty = defaultdict(list)
        
        for record in tqdm(data, desc="Processing outputs"):
            text = record["output"]["text"]
            field = record["field"]
            mean_sentence_certainty, aspect_scores = self.get_uncertainty_estimations(text)
            if field and mean_sentence_certainty and aspect_scores is not None:
                field_sentence_certainty[field].append(mean_sentence_certainty)
                field_aspect_certainty[field].append(aspect_scores)

        return dict(field_sentence_certainty), dict(field_aspect_certainty)


    # -------------------- Evaluation methods: ATTR -------------------- 
    # Assumes the citations are already parsed using the parse_citations module.

    def normalize_title(self, text):
        if not text:
            return ""

        # Normalize unicode accents
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

        # Lowercase
        text = text.lower()

        # Replace unicode dashes with normal hyphen
        text = re.sub(r"[–—−]", "-", text)

        # Remove punctuation
        text = re.sub(r"[^\w\s-]", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def title_similarity(self, a, b):
        """
        Returns fuzzy ration score (between 0 and 100).
        """
        a_norm = self.normalize_title(a)
        b_norm = self.normalize_title(b)

        return fuzz.ratio(a_norm, b_norm)
    
    def lookup_title_openalex(self, title, min_score=85):
        """
        Search for a paper title using OpenAlex with fuzzy matching.
        """

        # Query OpenAlex
        url = "https://api.openalex.org/works"
        params = {
            "search": title,
            "per-page": 25,
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            return {"exists": False, "error": str(e)}

        if not results:
            return {
                "exists": False,
                "score": 0,
                "openalex_id": None,
                "doi": None,
                "best_title": None,
                "metadata": None,
            }

        # Extract and normalize candidate titles
        candidate_titles = [res["title"] for res in results if "title" in res]

        # Use enhanced similarity scoring for each candidate
        best_score = -1
        best_title = None
        best_match = None

        for cand, match_obj in zip(candidate_titles, results):
            score = self.title_similarity(title, cand)

            if score > best_score:
                best_score = score
                best_title = cand
                best_match = match_obj

        exists = best_score >= min_score

        return {
            "exists": exists,
            "score": best_score,
            "openalex_id": best_match["id"] if exists else None,
            "doi": best_match.get("doi") if exists else None,
            "best_title": best_title if exists else None,
            "metadata": best_match if exists else None,
        }


    # -------------------- Save / export --------------------

    def save_results(self, out_path):
        """Save evaluation summaries to a structured JSON file."""
        to_save = {
            "model": self.model_name,
            "results": self.results,
        }
        with open(out_path, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"💾 Saved evaluation results for {self.model_name} to {out_path}")
