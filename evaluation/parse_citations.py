import argparse
import re
import requests
import json
from hashlib import md5
from pathlib import Path

def split_citations(text):
    """
    Split citations separated by semicolons, numbers, or bullet points (*,-,•).
    """

    # 1) Strip HTML links like <https://doi.org/...>
    text = re.sub(r"<https?://[^>]+>", "", text)

    # 2) Try semicolon-separated citations
    if ";" in text:
        parts = [c.strip() for c in text.split(";") if c.strip()]
        return parts

    # 3) Try bullet-point list (*, -, •)
    bullet_pattern = r"(?:\n|\r|^)\s*[*\-•]\s+"
    if re.search(bullet_pattern, text):
        # Split on the bullet, clean whitespace
        parts = re.split(bullet_pattern, text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    # 4) Try numbered list (1., 2., etc.)
    numbered = re.split(r"\n?\s*\d+\.\s+", text)
    numbered = [c.strip() for c in numbered if c.strip()]
    return numbered

def parse_citation(citation):
    """
    Parse a single APA citation into components.
    Returns dict with authors, year, title, venue, doi
    """

    parsed = {
        "authors": None,
        "year": None,
        "title": None,
        "venue": None,
        "doi": None
    }

    # --- DOI ---
    doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", citation)
    if doi_match:
        parsed["doi"] = doi_match.group(1)
        citation = citation.replace(doi_match.group(1), "")

    # --- Authors + Year ---
    # Accept:
    # (2014), (350 B.C.), (350 BC), (50 A.D.), (50 AD)
    year_pattern = r"^\s*(.*?)\s*\(\s*([0-9]{1,4})\s*(B\.?C\.?|A\.?D\.?)?\s*\)\.?\s*(.*)$"
    m = re.match(year_pattern, citation, flags=re.IGNORECASE)
    if not m:
        return parsed

    authors, year_num, era, remainder = m.groups()
    parsed["authors"] = authors.strip().rstrip(".")

    # Normalize year
    if era and "B" in era.upper():      # BC year
        parsed["year"] = f"{year_num} BC"
    elif era and "A" in era.upper():    # AD year (same as regular)
        parsed["year"] = f"{year_num} AD"
    else:                               # Normal year
        parsed["year"] = year_num

    # --- Title + Venue ---
    title_venue_match = re.match(r"^(.*?)\.\s+(.*)$", remainder)
    if title_venue_match:
        parsed["title"] = title_venue_match.group(1).strip()
        parsed["venue"] = title_venue_match.group(2).strip()
    else:
        parsed["title"] = remainder.strip()
        parsed["venue"] = None

    # OPTIONAL: Ignore some titles (your previous logic)
    if parsed['title'] and parsed['title'].lower() in [
        "alzheimer's disease in young adults",
        "individual proximity, complex and coordinated collective movement"
    ]:
        return None

    return parsed

def parse_apa_list(text):
    citations = split_citations(text)
    citations = list(set(citations))  # remove duplicate raw strings

    results = [parse_citation(c) for c in citations]
    results = [r for r in results if r is not None]

    # remove citations where ALL values are None
    filtered = [r for r in results if not all(v is None for v in r.values())]

    # remove duplicate dictionaries
    # convert to canonical string form & hash to detect duplicates
    unique = []
    seen = set()
    for r in filtered:
        # convert dict to sorted string so order doesn’t matter
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

def field_summary(citations_output):

    field_counter = {
        'Engineering': [],
        'Life Sciences': [],
        'Physical Sciences and Mathematics': [],
        'Social and Behavioural Sciences': [],
        'Arts and Humanities': [],
        'Total': []
    }

    for output in citations_output:
        field_counter[output['field']].append(output['citations_parsed'])
        field_counter['Total'].append(output['citations_parsed'])

    # make this a for loop 
    lengths = [len(claim_list) for claim_list in field_counter['Engineering']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Engineering'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")

    lengths = [len(claim_list) for claim_list in field_counter['Life Sciences']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Life Sciences'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")

    lengths = [len(claim_list) for claim_list in field_counter['Physical Sciences and Mathematics']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Physical Sciences and Mathematics'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")
    
    lengths = [len(claim_list) for claim_list in field_counter['Social and Behavioural Sciences']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Social and Behavioural Sciences'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")

    lengths = [len(claim_list) for claim_list in field_counter['Arts and Humanities']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Arts and Humanities'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")

    lengths = [len(claim_list) for claim_list in field_counter['Total']]
    average_length = sum(lengths) / len(lengths)
    print(f"The average length of lists in field_counter['Total'] is: {average_length}")
    print(f"Total citations for Engineering: {sum(lengths)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    save_path = Path(args.output_path)

    citations_filename = args.model_name + "_citations.jsonl"
    citations_path = save_path / citations_filename

    with open(input_path, 'r') as f:
        data = f.readlines()
        data = [json.loads(claim) for claim in data]
    
    citations_output = []
    for output in data:
        citation_text = output['output_citation']['text']
        citations = parse_apa_list(citation_text)
        citations_output.append({
            'id': output['id'],
            'field': output['field'],
            'entity': output['entity'],
            'citations_text': citation_text,
            'citations_parsed': citations,
            'logprobs': output['output_citation']['logprobs'],
            'meta': output['output_citation']['meta']
        })
    
    # save citations_output as jsonl
    with open(citations_path, 'w') as f:
        for item in citations_output:
            f.write(json.dumps(item) + '\n')

    field_summary(citations_output)

    


