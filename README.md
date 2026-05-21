# SciFactCheck

SciFactCheck is a benchmark and evaluation framework for assessing scientific factuality hallucinations in open-ended, closed-book LLM-generated text across five scientific domains: Engineering, Life Sciences, Physical Sciences and Mathematics, Social and Behavioural Sciences, and Arts and Humanities. The framework targets three hallucination types: unverifiability (VER), overclaim (OVR), and attribution (ATR), with dedicated automatic metrics per type. 

This repository contains the code for benchmark construction, model prompting, and evaluation.

## Repository Structure

```
SciFactCheck/
├── data_prep/          # Benchmark construction + utils
├── prompting/          # Model inference
└── evaluation/         # Hallucination evaluation
```

## 1. Benchmark Construction

The benchmark consists of 2,500 scientific concepts across five domains, grounded in peer-reviewed survey literature. Concepts are extracted semi-automatically from highly cited (>=100 citations) survey papers retrieved from S2ORC, and from the Internet Encyclopedia of Philosophy (IEP) for Arts and Humanities.

- **S2ORC survey paper retrieval:** `data_prep/s2orc_api.py`
- **IEP article scraping:** `data_prep/iep.py`
- **Full dataset:** 

## 2. Model Prompting

We use a two-step prompting strategy: first generating a scientific paragraph, then eliciting citations separately. 

- **Paragraph generation prompt:** `prompting/prompt_paragraph.txt`
- **Citation generation prompt:** `prompting/prompt_citation.txt`
- **Inference script:** `prompting/run_model.py`

The inference script supports three backends: HuggingFace, LiteLLM, and vLLM:

```bash
python prompting/run_model.py \
    --model <model_name_or_path> \
    --backend <hf|litelml|vllm> \
    --prompt_template <path_to_template> \
    --input <path_to_concepts_csv> \
    --output <path_to_output_jsonl>
```

**Sample output format**: one record per concept per model would look like this:
```json
{
    "id": 0,
    "field": "Engineering"
    "entity": "nanofluids",
    "two_prompts": true,
    "output": {
        "prompt": "Scientific claims are those that [...]",
        "text": "Nanofluids are engineered colloidal suspensions that [...]",
        "logprobs": [...],
        "length_norm_logprob": -0.6522063442125952,
    }
    "meta": {
        "model": "openai/gpt-4o-mini",
        "backend": "litellm",
        "num_tokens": 249,
    }
    "output_citation": {
        "prompt": "Find relevant scientific or academic [...]",
        "text": "Hassan, H. A., & Badran, M. A. (2016). [...]",
        "logprobs": [...],
        "length_norm_logprob": -219.88604030781042,
        "meta": {
            "model": "openai/gpt-4o-mini",
            "backend": "litellm",
            "num_tokens": 319
        }
    }
}
```

## 3. Evaluation Framework

### 3.1 Unverifiability Hallucination (VER)

We use two metrics with different knowledge sources:

**OpenFActScore (OFS)** verifies claims against the full text of the source survey paper.
- KB construction code: `data_prep/prepare_ofs_kb.py`
- We use a forked version of OpenFActScore: [link to fork]

**VeriScore** verifies claims against Google Search results via the Serper API.
- We use a forked version of VeriScore: [link to fork]

Both metrics use VeriScore's claim extraction model for consistency.

### 3.2 Overclaim Hallucination (OVR)

Two complementary uncertainty proxies are computed:

- **Linguistic uncertainty** (certainty estimator): sentence-level certainty scores
- **Parametric uncertainty** (log probabilities): length-normalized mean per-token log probability

```bash
python evaluation/run_eval.py \
    --task ovr \
    --input <path_to_responses_jsonl> \
    --output <path_to_results_dir>
```

### 3.3 Attribution Hallucination (ATR)

Citations are verified against CrossRef using DOI-based and title-based lookup.
A citation is deemed valid if either strategy confirms it.

- **CrossRef validator:** `evaluation/citation_validator.py`

```bash
python evaluation/run_eval.py \
    --task atr \
    --input <path_to_responses_jsonl> \
    --output <path_to_results_dir>
```

