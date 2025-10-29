"""
Core evaluation class for analyzing LLM-generated scientific outputs.

This module defines the `ModelEvaluator` class, which provides low-level
functions to analyze a model's outputs at both sample and aggregate levels.
It directly operates on nested dictionary data.

Main functionalities based on factuality hallucination type:

1. OVR 
-- Logprob evaluation: compute and summarize model confidence by field.
-- Linguistic uncertainty estimation: quantify certainty in generated text using a pretrained `CertaintyEstimator` model.

2. ATTR: check if cited papers are valid using the Semantic Scholar API.


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
            if sentences
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
    # Code adapted from https://github.com/AbhilashaRavichander/HALoGEN/blob/main/src/scientific_attribution.py 

    def _query_semantic_scholar(self, title):
       """
       Query the Semantic Scholar API for a given title.
       Args:
       - title (str): The title to query.
       
       Returns:
       - str: The title of the paper returned by the API, or an empty string if not found.
       """
       base_url = "https://api.semanticscholar.org/graph/v1/paper/"
       
       if not title or not isinstance(title, str):
          return None
       
       query_title = '+'.join(title.split())
       url = f"{base_url}search?query={query_title}&fields=url"
       
       time.sleep(1)  # Delay to avoid hitting rate limits
       response = requests.get(url)
       
       if response.status_code == 200:
          title_search_results = response.json()
          if 'data' in title_search_results and title_search_results['data']:
             paper_id = title_search_results['data'][0]['paperId']
             paper_url = f"{base_url}{paper_id}"
             
             retry_count = 0
             while retry_count < 3:
                 paper_response = requests.get(paper_url)
                 if paper_response.status_code == 200:
                     data = paper_response.json()
                     return data.get('title', '')
                 
                 elif paper_response.status_code == 429:
                    retry_count += 1
                    time.sleep(5)
                
                 else:
                     return ''
    
       return ''
    
    def extract_citations(self, llm_output: str):
        """
        Extracts the citations JSON block from an LLM output formatted according to the prompt template.
        **Note: if the JSON is not well-structured and cannot be read, the function will return None, and the sample will not be included in the evaluation.

        Args:
        llm_output (str): The model output that includes a JSON block of citations.
        Returns:
        list[dict] | None: The list of citations if correctly formatted, otherwise None.
        """
        # Match a JSON block starting with {"citations": [...]}
        match = re.search(r'\{\s*"citations"\s*:\s*\[.*?\]\s*\}', llm_output, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return None
        
        json_str = match.group(0)
        
        try:
            parsed = json.loads(json_str)
            citations = parsed.get("citations", None)
            
            # Return only if it's a list
            if isinstance(citations, list):
                return citations
            else:
                return None
            
        except json.JSONDecodeError as e:
            print("Error parsing citations JSON:", e)
            return None
        
    def _clean_text(self, text):
        
        if not isinstance(text, str) or text is None:
            return ""
        return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    
    def process_attribution_hallucinations(self, llm_output):
        """
        Given an output from an LLM, the function:
        1. extracts citations
        2. goes through them and for each one queries S2ORC based on title and saves the results
        3. checks if the citation from the LLM matches the one returned by S2ORC after cleaning
        
        Returns:
        - score: the number of hallucinated citations divided by the overall number of citations given by the LLM (percentage)
        """
        
        citations = self.extract_citations(llm_output)
        if citations is None or len(citations) == 0:
            return None
        
        hallucinated_citations = 0
        
        for citation in tqdm(citations, desc="Verifying citations against S2ORC"):
            s2orc_title = self._query_semantic_scholar(title=citation['title'])
            if self._clean_text(citation['title']) != self._clean_text(s2orc_title):
                hallucinated_citations += 1      
                
        score = hallucinated_citations / len(citations) * 100
        return score
    
    def collect_attr_by_field(self, data):
        """
        Goes over data samples (i.e. LLM outputs) and generates report of attribution hallucination score per field. 
        """
        field_attr = defaultdict(list)
        
        for record in tqdm(data, desc="Processing outputs"):
            text = record["output"]["text"]
            field = record["field"]
            attr_score = self.process_attribution_hallucinations(text)
            if field and attr_score is not None:
                field_attr[field].append(attr_score)
                
        # Print summary
        print("=== ATTR-score summary by field ===")
        for field, values in field_attr.items():
            print(f"{field:20s} | {len(values):3d} samples | mean = {mean(values):.8f}")
            
        return dict(field_attr)

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
