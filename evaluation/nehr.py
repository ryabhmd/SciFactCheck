from keybert.llm import TextGeneration
from keybert import KeyLLM

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import string
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

import time
import tqdm
import Dict

def keybert_llm_sets(llm_output:str, fulltext_sections:Dict, model_name="CohereLabs/c4ai-command-a-03-2025"):
    """
    Args:
    - llm_output: the llm output that is being evaluated
    - fulltext_sections: the fulltext of the original survey paper in a dictionary structure such as:
    {
    'Introduction': [paragraph1, paragraph2, ...],
    'Related Work': [paragraph1, paragraph2, ...],
    ...
    }
    - model_name: the LLM to be used for extracting keywords as written on HF; defaults to CohereLabs/c4ai-command-a-03-2025

    The function uses keybert's KeyLLM feature using text generation to extract keywords from the llm output and from the fulltext.
    It returns:
    - set of keywords from the llm output.
    - set of keywords from the fulltext. 
    """

    generator = pipeline('text-generation', model=model_name)
    llm = TextGeneration(generator)
    kw_model = KeyLLM(llm)

    # Extract keywords from llm output
    keywords_llm = kw_model.extract_keywords(llm_output)
    keywords_llm = set(keywords_llm[0])
    with open('keywords_llm.txt', 'w') as f:
        f.write(str(keywords_llm))

    # Extract keywords from fulltext per paragraph
    keywords_fulltext = set()
    
    for key, value in tqdm(fulltext_sections.items()):
       for paragraph in value:
          keywords = kw_model.extract_keywords(paragraph)
          for keyword in keywords[0]:
             keywords_fulltext.add(keyword)
             # save results of keywords_fulltext
             with open('keywords_fulltext.txt', 'w') as f:
                 f.write(str(keywords_fulltext))

    return keywords_llm, keywords_fulltext

def chunk_text(text, tokenizer, chunk_size=510, overlap=128):
    """
    Splits text into overlapping chunks based on token length.

    Args:
        text (str): The input text.
        tokenizer: A Hugging Face tokenizer.
        chunk_size (int): Max tokens per chunk (≤510 for BERT-like models).
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    # Tokenize without truncation or special tokens
    tokens = tokenizer(
        text, return_tensors="pt", truncation=False, add_special_tokens=False
    )["input_ids"][0]

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk_ids = tokens[i:i + chunk_size]
        if len(chunk_ids) == 0:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + chunk_size >= len(tokens):  # stop if we’re at the end
            break

    return chunks


def scibert_ner_sets(llm_output:str, 
                     fulltext_sections, 
                     model_name="roccas98/Scibert_finetuned_scierc",
                     tokenizer_name="allenai/scibert_scivocab_uncased",
                     chunk_size=510,
                     overlap=128):
    """
    Args:
    - llm_output: the llm output that is being evaluated
    - fulltext_sections: the fulltext of the original survey paper in a dictionary structure such as:
    {
    'Introduction': [paragraph1, paragraph2, ...],
    'Related Work': [paragraph1, paragraph2, ...],
    ...
    }

    The function uses the HF NER pipeline and a fine-tuned model to extract NE from texts.
    It returns:
    - set of keywords from the llm output.
    - set of keywords from the fulltext. 
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Get NE from LLM output
    
    llm_output_chunks = chunk_text(llm_output, tokenizer, chunk_size=chunk_size, overlap=overlap)

    entities = []
    for chunk in llm_output_chunks:
        entities.extend(ner_pipeline(chunk))

    entities_llm = set()
    for entity in entities:
      entities_llm.add(entity['word'])

    # Get NE from fulltext

    entity_set_full_text = set()
    for key, value in tqdm.tqdm(fulltext_sections.items()):
       for paragraph in value:
          paragraph_chunks = chunk_text(paragraph, tokenizer, chunk_size=chunk_size, overlap=overlap)
          for chunk in paragraph_chunks:
             chunk_entities = ner_pipeline(chunk)
             for entity in chunk_entities:
                 entity_set_full_text.add(entity['word'])

    return entities_llm, entity_set_full_text


def nehr(llm_set, fulltext_set):
    """
    Computes NEHR (Named Entity Hallucination Risk) as defined in Akani et al. (2023)
    Args:
    - llm_set: the set of keywords or NE extracted from the llm output to be evaluated
    - fulltext_set: the set of keywords or NE extracted from original survey full text

    Returns NEHR score. 
    """
    llm_set_lowered = {s.lower() for s in llm_set}
    fulltext_set_lowered = {s.lower() for s in fulltext_set}

    intersection = llm_set_lowered.intersection(fulltext_set_lowered)

    nehr = (1- (len(intersection) / len(llm_set_lowered))) * 100

    return nehr








