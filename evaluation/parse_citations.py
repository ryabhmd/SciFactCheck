"""
    A utility class for parsing APA-style citations extracted from model outputs.

    Features:
      - Parse and clean individual citations (author, year, title, venue, DOI).
      - Split lists of citations from raw text (semicolon, bullet, numbered).
      - Parse entire JSONL files containing model inference outputs.
      - Save parsed citation results back to JSONL.
      - Print summary statistics by scientific field.

    Parameters
    ----------
    remove_italics : bool, optional (default=True)
        Whether to remove Markdown/APA italic markers (*...*).
    strip_doi_fragments : bool, optional (default=True)
        Whether to remove URL DOI prefixes like "https://doi.org/".
"""
import re
import json
from hashlib import md5
from pathlib import Path
from statistics import mean, median
import unicodedata

class CitationParser:

    def __init__(self, remove_italics=True, strip_doi_fragments=True):
        self.remove_italics = remove_italics
        self.strip_doi_fragments = strip_doi_fragments

    # -------------------------------------------------------
    # Cleaning utilities
    # -------------------------------------------------------
    def clean_citation_text(self, text: str) -> str:
        """
        Normalize and clean citation text components (title, venue).

        Operations performed:
        - Normalize unicode to NFKC
        - Remove italics markers (*, **)
        - Remove DOI URL prefixes but keep actual DOI values
        - Collapse extra whitespace

        Parameters:
        - text : str
            Raw title or venue string from a citation.

        Returns:
        - str: A cleaned version of the text.
        """

        if not text:
            return text

        t = unicodedata.normalize("NFKC", text)

        # Remove italics markers (*, **)
        if self.remove_italics:
            t = re.sub(r"\*+", "", t)

        # Remove bare DOI prefixes like https://doi.org/ but keep actual DOI number
        if self.strip_doi_fragments:
            t = re.sub(r"https?://doi\.org/?", "", t, flags=re.IGNORECASE)
            t = re.sub(r"\bdoi:\s*", "", t, flags=re.IGNORECASE)

        return re.sub(r"\s+", " ", t).strip()

    # -------------------------------------------------------
    # Splitting utilities
    # -------------------------------------------------------
    def split_citations(self, text):
        """
        Split a string containing potentially multiple citations into a list.
        Discretion is advised: not all models stick to the desired structure and the function can be changed according to the model.

        Supported patterns include:
        - Semicolon-separated citations (e.g. "A (2020); B (2021)")
        - Bullet lists (*, -, •)
        - Numbered lists (1., 2., etc.)
        - Removal of <https://doi.org/...> link elements

        Parameters
        - text : str
            Raw citation text extracted from a model output.

        Returns
        - list[str]
            A list of raw citation strings.
        """
        text = re.sub(r"<https?://[^>]+>", "", text)  # remove links

        # Try semicolon-separated citations
        if ";" in text:
            return [c.strip() for c in text.split(";") if c.strip()]

        # Bullet list (*, -, •)
        bullet_pattern = r"(?:\n|\r|^)\s*[*\-•]\s+"
        if re.search(bullet_pattern, text):
            parts = re.split(bullet_pattern, text)
            return [p.strip() for p in parts if p.strip()]

        # Numbered list (1., 2., etc.)
        numbered = re.split(r"\n?\s*\d+\.\s+", text)
        return [c.strip() for c in numbered if c.strip()]

    # -------------------------------------------------------
    # Core parser
    # -------------------------------------------------------
    def parse_citation(self, citation):
        """
        Parse a single APA-style citation into structured components.

        Extracted fields:
        - authors
        - year (supports BC/AD formats)
        - title
        - venue (journal, conference, book)
        - doi

        Parameters
        - citation : str
            A raw citation string.

        Returns
        - dict or None
            Dictionary representation of the citation, or None if invalid or ignored.
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
            found_doi = doi_match.group(1).strip()
            if found_doi.lower() in {"10.1167/com0000274"}:  # ignore
                return None
            parsed["doi"] = found_doi
            citation = citation.replace(found_doi, "")

        # --- Authors + Year ---
        year_pattern = r"^\s*(.*?)\s*\(\s*([0-9]{1,4})\s*(B\.?C\.?|A\.?D\.?)?\s*\)\.?\s*(.*)$"
        m = re.match(year_pattern, citation, flags=re.IGNORECASE)
        if not m:
            return parsed

        authors, year_num, era, remainder = m.groups()
        parsed["authors"] = authors.strip().rstrip(".")

        # Normalize year
        if era and "B" in era.upper():
            parsed["year"] = f"{year_num} BC"
        elif era and "A" in era.upper():
            parsed["year"] = f"{year_num} AD"
        else:
            parsed["year"] = year_num

        # --- Title + Venue ---
        title_venue_match = re.match(r"^(.*?)\.\s+(.*)$", remainder)
        if title_venue_match:
            parsed["title"] = title_venue_match.group(1).strip()
            parsed["venue"] = title_venue_match.group(2).strip()
        else:
            parsed["title"] = remainder.strip()

        if parsed["title"]:
            parsed["title"] = self.clean_citation_text(parsed["title"])

        if parsed["venue"]:
            parsed["venue"] = self.clean_citation_text(parsed["venue"])
            
        # Ignore titles given in prompt as examples
        if parsed['title'] and parsed['title'].lower() in [
            "alzheimer's disease in young adults",
            "individual proximity, complex and coordinated collective movement"
            ]:
            return None

        return parsed

    # -------------------------------------------------------
    # Batch parsing (per one model response)
    # -------------------------------------------------------
    def parse_apa_list(self, text):
        """
        Parse multiple APA citations from a text block and deduplicate them.

        Steps:
        - Split citation text into individual entries.
        - Parse each citation into structured components.
        - Remove entries where all fields are None.
        - Deduplicate citations by structural content.

        Parameters
        - text : str
            Raw multiline citation block.

        Returns
        - list[dict]
            List of parsed citation dictionaries.
        """
        # Remove <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
         
        citations = self.split_citations(text)
        citations = list(set(citations))  # remove duplicate raw strings

        results = [self.parse_citation(c) for c in citations]
        results = [r for r in results if r is not None]
        filtered = [r for r in results if not all(v is None for v in r.values())]

        # remove duplicate dictionaries
        unique, seen = [], set()
        for r in filtered:
            key = json.dumps(r, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique
    
    # -------------------------------------------------------
    # Read model results and parse citations for all prompts
    # -------------------------------------------------------
    def run(self, input_path: Path, citations_path: Path):
        """
        Read a JSONL file containing model outputs, parse citations for each entry,
        and save structured results to a new JSONL file.

        Expected input format is the result of the prompting module. 
        Each line must be a JSON object containing:
            - id
            - field
            - entity
            - output_citation.text
            - output_citation.logprobs
            - output_citation.meta

        JSONL output structure:
        {
            'id', 'field', 'entity', 'citations_text',
            'citations_parsed', 'logprobs', 'meta'
        }

        Parameters
        - input_path : Path
            Path to the JSONL file containing model outputs.
        - citations_path : Path
            Where to save the parsed citation JSONL.

        Returns
        - list[dict]
            The parsed results loaded into memory.
        """
        with open(input_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        citations_output = []
        for output in data:
            citation_text = output['output_citation']['text']
            citations = self.parse_apa_list(citation_text)
            citations_output.append({
                'id': output['id'],
                'field': output['field'],
                'entity': output['entity'],
                'citations_text': citation_text,
                'citations_parsed': citations,
                'logprobs': output['output_citation']['logprobs'],
                'meta': output['output_citation']['meta']
            })

        # Save as JSONL
        with open(citations_path, 'w') as f:
            for item in citations_output:
                f.write(json.dumps(item) + '\n')

        return citations_output
    
    # -------------------------------------------------------
    # Print summary by field
    # -------------------------------------------------------
    def field_summary(self, citations_output: list):
        """
        Print statistical summary of the number of parsed citations per field.

        Parameters
        ----------
        citations_output : list
            A list of citation result dictionaries, typically returned from `run()`.

        Prints
        ------
        For each field:
            - number of outputs
            - total citations
            - average citations/output
            - median citations/output
        """
        per_field = {}

        # Count citations per field
        for item in citations_output:
            field = item['field']
            count = len(item['citations_parsed'])

            if field not in per_field:
                per_field[field] = []
            per_field[field].append(count)

        # Print summary statistics
        for field, counts in per_field.items():
            total_entries = len(counts)
            total_citations = sum(counts)
            avg_citations = mean(counts)
            med_citations = median(counts)

            print(f"Field: {field}")
            print(f"  # Outputs: {total_entries}")
            print(f"  Total citations: {total_citations}")
            print(f"  Average citations/output: {avg_citations:.2f}")
            print(f"  Median citations/output: {med_citations:.2f}")
            print("-" * 40)


    


