import ast
import re
import time
from tqdm import tqdm

from google import genai
from google.genai import types

from sentence_transformers import SentenceTransformer


def parse_triples(text):
    """
    Convert a string containing ```python fenced list``` into a real Python list.
    """
    # Remove ```python ... ``` fencing if present
    cleaned = re.sub(r"^```python\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    # Safely evaluate the remaining string into a Python object
    return ast.literal_eval(cleaned)


def get_gemini_triples(text:str, GOOGLE_API_KEY:str, gemini_model="gemini-2.5-flash", system_prompt_path="kg-extract-system-prompt.txt", ):
    """
    A function that uses Google's GenAI to prompt for triple extraction from raw text. 
    Args: 
    - text: text to extract triple from
    - GOOGLE_API_KEY: api key to access Gemini
    - gemini_model: exact model to use, default is gemini_model
    - system_prompt_path: path to extract system prompt.

    Returns a list of triples; each triple is a list of [subject, predicate, object]
    """

    with open(system_prompt_path, 'r') as f:
        kg_extract_system_prompt = f.read()

    kg_extract_system_prompt 

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=kg_extract_system_prompt),
            contents=f"Text: {text}"
            )
    
    triples = parse_triples(response.text)

    return triples

def get_scibert_triples():
    """
    TODO
    """

    return None


def rehr(llm_triples, fulltext_triples, embedding_model="allenai/scibert_scivocab_cased", threshold=0.8):
  """
  Two relation triplets are considered to be equivalent if their embeddings have cosine similarity higher than a threshold for all three elements of the triplet.
  The function should go over each triple created from the llm output and check if there is at least one triple in the fulltext that matches at least 0.6 cosine similarity in all of its elements (s, p, o).
  If one triple is found -> no need to check the next ones -> the llm triple is marked as True.
  Once the function iterates over all llm triples, it computed the REHR score (inspired by NEHR):
  (1 - (triples found in fulltext / total llm triples)) * 100
  """

  model = SentenceTransformer(embedding_model)

  # compute embeddings of fulltext_triples and save them
  fulltext_triples_embeddings = []
  for triple in tqdm(fulltext_triples, desc="Computing fulltext triples embeddings"):
    embeddings = model.encode(triple)
    fulltext_triples_embeddings.append(embeddings)

  found_flags = []

  for llm_triple in tqdm(llm_triples, desc="Iterating over LLM triples"):

    triple_found = False

    for i, fulltext_triple in enumerate(fulltext_triples):

      all_match = True

      for j in range(3):
        llm_triple_embeddings = model.encode([llm_triple[j]])
        cosine_similarity = float(model.similarity(llm_triple_embeddings, fulltext_triples_embeddings[i][j])[0])
        print(f"Cosine similarity between {llm_triple[j]} and {fulltext_triple[j]}: {cosine_similarity}")

        if cosine_similarity < threshold:
          all_match = False
          print('Triple did not meet threshold.')
          break

      if all_match:
        triple_found = True
        break

    found_flags.append(triple_found)

  # compute REHR
  found_count = sum(found_flags)
  total = len(llm_triples)
  rehr_score = (1 - found_count / total) * 100

  return rehr_score


def main():
   """
   TODO: function to read llm output + fulltext, extract triples from them (either using gemini or scibert), and computre rehr score
   """