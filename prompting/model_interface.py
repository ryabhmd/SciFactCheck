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

    def __init__(self, model_id: str, api_key: Optional[str] = None, seed: int = 0, force_download: Optional[bool] = True, apply_chat_template: Optional[bool] = False):
        from transformers import (
        AutoModelForCausalLM, AutoTokenizer
        )

        self.model_id = model_id
        torch.manual_seed(seed)
        self.api_key = api_key or os.getenv("HF_API_KEY")
        self.apply_chat_template = apply_chat_template
        
        if "galactica" in self.model_id:
            from transformers import OPTForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.api_key)
            self.model = OPTForCausalLM.from_pretrained(
                model_id, 
                token=self.api_key,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                force_download=force_download
            )

        elif "SciDFM" in self.model_id:
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id, use_fast=False, token=self.api_key)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                token=self.api_key,
                torch_dtype=torch.bfloat16, 
                device_map="auto", 
                trust_remote_code=True
            )

        elif "forge" in self.model_id:
            from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.model_id, token=self.api_key)
            self.model = GPTNeoXForCausalLM.from_pretrained(
                model_id,
                token=self.api_key,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                force_download=force_download
            )
    
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.api_key, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.api_key,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self.device = self.model.device
        logging.info(f"Loaded HuggingFace model: {model_id} on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> Dict[str, Any]:

        if "SciDFM" in self.model_id:
            chat_template = "<|user|>:{instruction}<|assistant|>:"
            prompt = chat_template.format(instruction=prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        else:
            if self.apply_chat_template:
                messages = [
                {"role": "user", "content": prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True, 
                    enable_thinking=False
                )
            else:
                prompt = prompt + " **Answer:**"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.1,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
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
            "prompt": prompt,
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

    def __init__(self, model_name: str, api_key: Optional[str] = None, gemini_key: Optional[str] = None):
        from litellm import completion
        self.completion = completion
        self.model_name = model_name
        #self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        #self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

        #if not self.api_key:
        #    logging.warning(
        #        f"No API key found for {model_name}. Make sure environment vars are set."
        #    )

        logging.info(f"Loaded LiteLLM model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> Dict[str, Any]:
        try:
            response = self.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
               # max_completion_tokens=max_tokens,
               # logprobs=True,  # Request logprobs if supported
               # **kwargs,
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
            "prompt": prompt,
            "text": text,
            "logprobs": logprobs,
            "length_norm_logprob": length_norm_logprob,
            "meta": {
                "model": self.model_name,
                "backend": "litellm",
                "num_tokens": len(logprobs) if logprobs else None,
            },
        }
    
class vLLMModel(BaseModelInterface):
    """Local vLLM model wrapper with chat templating and logprobs."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = False,
        seed: int = 0,
        is_chat: bool = False,
        return_logprobs: bool = True,
    ):
        from vllm import LLM

        self.model_name = model_name
        self.is_chat = is_chat
        self.return_logprobs = return_logprobs

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )

        logging.info(
            f"Loaded vLLM model: {model_name} "
            f"(tp={tensor_parallel_size}, chat={is_chat}, logprobs={return_logprobs})"
        )

    def _build_chat_prompt(self, user_prompt: str):
        return  [
            {"role": "user", "content": user_prompt},
            ]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            logprobs=1 if self.return_logprobs else None,
        )

        # ---------------------------
        # Generate
        # ---------------------------
        if self.is_chat:
            conversation = self._build_chat_prompt(prompt)
            outputs = self.llm.chat(
                conversation,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            used_prompt = conversation
        else:
            if "tulu" in self.model_name:
                prompt = ("<|user|>\n" + prompt + "<|assistant|>\n")

            outputs = self.llm.generate(
                prompt,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            used_prompt = prompt

        output = outputs[0].outputs[0]
        generated_text = output.text

        # ---------------------------
        # Extract logprobs
        # ---------------------------
        logprobs = None
        length_norm_logprob = None

        if self.return_logprobs and output.logprobs:
            logprobs = []
            for token_lp in output.logprobs:
                # token_lp is a dict where values are Logprob objects
                logprob_obj = next(iter(token_lp.values()))

                # Check if it's an object (vLLM >= 0.4.x) or just a float
                if hasattr(logprob_obj, 'logprob'):
                    logprobs.append(logprob_obj.logprob)
                else:
                    logprobs.append(logprob_obj)

            if logprobs:
                length_norm_logprob = sum(logprobs) / len(logprobs)

        return {
            "prompt": used_prompt,
            "text": generated_text,
            "logprobs": logprobs,
            "length_norm_logprob": length_norm_logprob,
            "meta": {
                "model": self.model_name,
                "backend": "vllm",
                "num_tokens": len(output.token_ids),
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
