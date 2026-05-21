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

The inference script supports three backends: HuggingFace (`hf`), LiteLLM (`litellm`), and vLLM (`vllm`). API keys can be passed as arguments or set as environment variables (`HF_API_KEY`, `LITELLM_API_KEY`).

### Basic usage (single prompt)

```bash
python prompting/run_model.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type hf \
    --dataset_name hf_dataset \
    --prompt_template prompting/prompt_paragraph.txt \
    --output_dir results/ \
    --max_new_tokens 512 \
    --apply_chat_template
```

### Two-prompt strategy (paragraph + citations)

```bash
python prompting/run_model.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type hf \
    --dataset_name hf_dataset \
    --prompt_template prompting/prompt_paragraph.txt \
    --two_prompts \
    --citation_prompt_template prompting/prompt_citation.txt \
    --output_dir results/ \
    --apply_chat_template
```

### Resuming from a checkpoint

```bash
python prompting/run_model.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_type vllm \
    --tensor_parallel_size 4 \
    --dataset_name hf_dataset \
    --prompt_template prompting/prompt_paragraph.txt \
    --two_prompts \
    --citation_prompt_template prompting/prompt_citation.txt \
    --from_checkpoint \
    --checkpoint_path results/Qwen_Qwen2.5-7B-Instruct_checkpoint.jsonl \
    --output_dir results/
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_name` | ✓ | — | Model name or HuggingFace path |
| `--model_type` | ✓ | — | Backend: `hf`, `litellm`, or `vllm` |
| `--dataset_name` | ✓ | — | HuggingFace dataset identifier (SciFactCheck) |
| `--prompt_template` | ✓ | — | Path to `.txt` prompt template; use `{scientific_entity}` and `{field}` as placeholders |
| `--output_dir` | | `results/` | Directory to save outputs |
| `--max_new_tokens` | | `512` | Maximum number of tokens to generate |
| `--tensor_parallel_size` | | `1` | Number of tensor parallel partitions (vLLM only) |
| `--apply_chat_template` | | `False` | Format prompts with `tokenizer.apply_chat_template()` |
| `--two_prompts` | | `False` | Use two-prompt strategy (paragraph + citations) |
| `--citation_prompt_template` | | — | Path to citation prompt template (required if `--two_prompts`) |
| `--from_checkpoint` | | `False` | Resume from a previous checkpoint |
| `--checkpoint_path` | | — | Path to checkpoint file (required if `--from_checkpoint`) |
| `--limit` | | `None` | Limit number of examples (for testing) |
| `--hf_api_key` | | `$HF_API_KEY` | HuggingFace API key |
| `--litellm_api_key` | | `$LITELLM_API_KEY` | LiteLLM API key |

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


### 3.3 Attribution Hallucination (ATR)

Citations are verified against CrossRef using DOI-based and title-based lookup.
A citation is deemed valid if either strategy confirms it.

- **CrossRef validator:** `evaluation/citation_validator.py`

OVR and ATR metrics can be run directly using `evaluation/run_eval.py`.

### Basic usage

```bash
python evaluation/run_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --file results/Qwen_Qwen2.5-7B-Instruct.jsonl \
    --save results/eval/ \
    --eval ovr-uncertainty ovr-logprobs attr
```

### With citation parsing

If your model output contains raw citation strings that need to be parsed 
before ATR evaluation, add `--parse_citations`:

```bash
python evaluation/run_eval.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --file results/Qwen_Qwen2.5-7B-Instruct.jsonl \
    --save results/eval/ \
    --eval atr \
    --parse_citations
```

### All arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_name` | ✓ | — | Model name or HuggingFace path |
| `--file` | ✓ | — | Path to model output JSONL file |
| `--save` | ✓ | — | Directory to save evaluation results |
| `--eval` | ✓ | — | One or more metrics to evaluate: `ovr-uncertainty` (linguistic uncertainty), `ovr-logprobs` (parametric uncertainty), `attr` (attribution validation) |
| `--parse_citations` | | `False` | Parse raw citation strings before ATR evaluation |

**Expected output format** (`results/<model_name>.jsonl`): one record per concept:
```json
{
    "id": 0,
    "field": "Engineering",
    "entity": "nanofluids",
    "two_prompts": true,
    "output": {
        "prompt": "...",
        "text": "...",
        "logprobs": [...],
        "length_norm_logprob": -0.652,
        "meta": {
            "model": "openai/gpt-4o-mini",
            "backend": "litellm",
            "num_tokens": 249
        }
    },
    "output_citation": {
        "prompt": "...",
        "text": "...",
        "logprobs": [...],
        "length_norm_logprob": -219.88,
        "meta": {
            "model": "openai/gpt-4o-mini",
            "backend": "litellm",
            "num_tokens": 319
        },
        "parsed_citations": [
            {
                "authors": "Choi, S. U. S., & Eastman, J. A",
                "year": "1995",
                ...
            }
        ]
    },
    "eval_logprobs": -0.652,
    "eval_uncertainty": {
        "mean_sentence_certainty": 4.77,
        "aspect_scores": {
            "Probability": 0.67,
            "Number": 0.0,
            "Extent": 0.0,
            "Condition": null,
            "Framing": 1.0,
            "Suggestion": 1.0
        }
    }
}
```
