"""
Unified interface for prompting and evaluating LLMs with LiteLLM + HuggingFace.

Supports:
- Any API model available via LiteLLM (OpenAI, Anthropic, Gemini, Mistral, etc.)
- Local Hugging Face models (Galactica, Llama, Qwen, Gemma, etc.)

Returns a consistent schema:
{
    "text": str,
    "logprobs": list[float] or None,
    "length_norm_logprob": float or None,
    "meta": dict
}
"""

import os
import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment vars if .env exists
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- Base Interface ----------
class BaseModelInterface(ABC):
    """Abstract base class for any language model."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass


# ---------- HuggingFace Model ----------
class HFModel(BaseModelInterface):
    """Local HuggingFace model with detailed logprobs."""

    def __init__(self, model_id: str, api_key: Optional[str] = None, seed: int = 0, force_download: Optional[bool] = True):
        from transformers import (
        AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer
        )

        self.model_id = model_id
        torch.manual_seed(seed)
        self.api_key = api_key or os.getenv("HF_API_KEY")
        
        if "galactica" in self.model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_auth_token=self.api_key)
            self.model = OPTForCausalLM.from_pretrained(
                model_id, 
                use_auth_token=self.api_key,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                force_download=force_download
            )

        elif "SciDFM" in self.model_id:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id, use_fast=False, use_auth_token=self.api_key)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                use_auth_token=self.api_key,
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
                trust_remote_code=True
            )
    
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=self.api_key)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_auth_token=self.api_key,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                force_download=force_download
            )
        self.device = self.model.device
        logging.info(f"Loaded HuggingFace model: {model_id} on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> Dict[str, Any]:

        if "galactica" in self.model_id:
            prompt = prompt + " [START_REF]"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True
            )

        elif "SciDFM" in self.model_id:
            chat_template = "<|user|>:{instruction}<|assistant|>:"
            prompt = chat_template.format(instruction=prompt)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            
        # Decode only generated continuation
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Compute per-token logprobs
        logprobs = []
        if outputs.scores:
            for i, scores in enumerate(outputs.scores):
                log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                token_id = outputs.sequences[0, inputs.input_ids.shape[1] + i]
                logprobs.append(log_probs[0, token_id].item())

        length_norm_logprob = (
            sum(logprobs) / len(logprobs) if len(logprobs) > 0 else None
        )

        return {
            "text": generated_text,
            "logprobs": logprobs,
            "length_norm_logprob": length_norm_logprob,
            "meta": {
                "model": self.model_id,
                "backend": "huggingface",
                "num_tokens": len(logprobs),
            },
        }


# ---------- LiteLLM Wrapper ----------
class LiteLLMModel(BaseModelInterface):
    """Generic API model via LiteLLM (OpenAI, Anthropic, Gemini, etc.)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from litellm import completion
        self.completion = completion
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")

        if not self.api_key:
            logging.warning(
                f"No API key found for {model_name}. Make sure environment vars are set."
            )

        logging.info(f"Loaded LiteLLM model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        try:
            response = self.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,  # Request logprobs if supported
                api_key=self.api_key,
                **kwargs,
            )
        except Exception as e:
            logging.error(f"LiteLLM error for {self.model_name}: {e}")
            return {"error": str(e), "meta": {"model": self.model_name}}

        choice = response["choices"][0]
        text = choice["message"]["content"]

        # Extract logprobs (if supported)
        logprobs = None
        length_norm_logprob = None
        try:
            token_logprobs = choice.get("logprobs", {}).get("content", [])
            if token_logprobs:
                logprobs = [t["logprob"] for t in token_logprobs if "logprob" in t]
                if logprobs:
                    length_norm_logprob = sum(logprobs) / len(logprobs)
        except Exception:
            pass  # Fallback: not all providers expose logprobs

        return {
            "text": text,
            "logprobs": logprobs,
            "length_norm_logprob": length_norm_logprob,
            "meta": {
                "model": self.model_name,
                "backend": "litellm",
                "num_tokens": len(logprobs) if logprobs else None,
            },
        }


# ---------- Model Manager ----------
class ModelManager:
    """Run multiple model interfaces with unified output schema."""

    def __init__(self, models: Dict[str, BaseModelInterface]):
        self.models = models

    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        results = {}
        for name, model in self.models.items():
            logging.info(f"Running model: {name}")
            try:
                results[name] = model.generate(prompt, **kwargs)
            except Exception as e:
                logging.error(f"Error running {name}: {e}")
                results[name] = {"error": str(e)}
        return results
