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

   - Code for preparing survey papers for KB construction is under data_prep
   - Link to forked repo

3. VeriScore

   - Link to forked repo

### Overclaim Hallucinations

### Attribution Hallucinations

