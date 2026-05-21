# SciFactCheck

## Creating the Benchmark

- Survey paper retrieval is under data_prep/s2orc_api.py
- IEP scraping is under data_prep/iep.py
- Full data ==> upload here

## Prompting Models

- Paragraph generation prompt template: prompting/prompt_paragraph.txt
- Citation generation prompt template: prompting/prompt_citation.txt
- Model inference framework is under prompting/run_model.py; connects to three platforms: HF, LiteLLM, vLLM

## Evaluation Framework

### Unverifiability Hallucinations

1. OpenFActScore (OFS)

2. VeriScore

### Overclaim Hallucinations

### Attribution Hallucinations

