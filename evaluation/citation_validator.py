import argparse
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests

DEFAULT_USER_AGENT = "raia-citation-validator/0.1 (mailto:nikolas.rauscher@dfki.de)"
MIN_TITLE_SCORE = 0.8
EXCLUDED_TITLE_PHRASES = {
    "alzheimer s disease in young adults",
    "individual proximity complex and coordinated collective movement",
}
EXCLUDED_DOIS = {"10.1167/com0000274"}
EXCLUDED_DOI_SUFFIXES = {"com0000274"}
SEARCH_SETTINGS_KEYS = {"max_results", "min_title_score"}
SEARCH_RESPONSE_KEYS = {
    "count",
    "page",
    "per_page",
    "total_results",
    "items_per_page",
    "query_used",
}


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def build_request_url(url: str, params: Optional[Dict[str, Any]] = None) -> str:
    return requests.Request("GET", url, params=params).prepare().url


def get_with_retries(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    max_retries: int = 3,
    backoff: float = 0.75,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                return {"status": "error", "error": str(exc), "response": None}
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else backoff
            time.sleep(delay)
            backoff *= 2
            continue
        if resp.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
            time.sleep(backoff)
            backoff *= 2
            continue
        return {"status": "ok", "response": resp}
    return {"status": "error", "error": "exhausted retries", "response": None}


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"text": resp.text}


def build_request_error(
    error: Optional[str],
    request_url: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "status": "error",
        "error": error,
        "raw": None,
        "http_status": None,
        "request_url": request_url,
    }
    if extra:
        result.update(extra)
    return result


def build_response_result(
    resp: requests.Response,
    request_url: str,
    *,
    ok_statuses: Tuple[int, ...] = (200,),
    not_found_statuses: Tuple[int, ...] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_payload = safe_json(resp)
    status_code = resp.status_code
    if status_code in ok_statuses:
        status = "found"
        error = None
    elif status_code in not_found_statuses:
        status = "not_found"
        error = None
    else:
        status = "error"
        error = f"status {status_code}"
    result = {
        "status": status,
        "raw": raw_payload,
        "http_status": status_code,
        "request_url": request_url,
    }
    if error:
        result["error"] = error
    if extra:
        result.update(extra)
    return result


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def parse_year(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    match = re.search(r"\d{4}", str(raw))
    return int(match.group(0)) if match else None


def surname_from_name(name: str) -> Optional[str]:
    if not name:
        return None
    raw = str(name)
    if "," in raw:
        raw = raw.split(",", 1)[0]
    norm = normalize_text(raw)
    if not norm:
        return None
    parts = norm.split()
    for token in reversed(parts):
        if len(token) > 1:
            return token
    return None


def parse_provided_authors(value: Optional[str]) -> List[str]:
    if not value:
        return []
    cleaned = value.replace("et al", "")
    pieces = re.split(r"&| and |;", cleaned)
    surnames: List[str] = []
    for piece in pieces:
        for segment in piece.split(","):
            name = surname_from_name(segment)
            if name and len(name) > 1:
                surnames.append(name)
    return surnames


def surnames_from_list(names: Iterable[str]) -> List[str]:
    found = []
    for name in names:
        surname = surname_from_name(name)
        if surname:
            found.append(surname)
    return found


def author_overlap(provided: List[str], candidates: List[str]) -> Tuple[int, int, int]:
    provided_set = set(provided)
    candidate_set = set(candidates)
    return len(provided_set & candidate_set), len(provided_set), len(candidate_set)


def compute_match_metrics(
    citation: Dict[str, Any],
    title: Optional[str],
    venue: Optional[str],
    authors: List[str],
    year: Optional[int],
) -> Dict[str, Any]:
    provided_title = normalize_text(citation.get("title"))
    candidate_title = normalize_text(title)
    title_score = similarity(provided_title, candidate_title) if provided_title and candidate_title else None

    provided_venue = normalize_text(citation.get("venue"))
    candidate_venue = normalize_text(venue)
    venue_score = similarity(provided_venue, candidate_venue) if provided_venue and candidate_venue else None

    provided_authors = parse_provided_authors(citation.get("authors"))
    candidate_authors = surnames_from_list(authors)
    overlap, provided_count, candidate_count = author_overlap(provided_authors, candidate_authors)
    author_fraction = overlap / provided_count if provided_count else None

    cited_year = parse_year(citation.get("year"))
    year_match = cited_year == year if cited_year and year else None

    parts: List[float] = []
    if title_score is not None:
        parts.append(title_score)
    if venue_score is not None:
        parts.append(venue_score)
    if author_fraction is not None:
        parts.append(author_fraction)
    if year_match is not None:
        parts.append(1.0 if year_match else 0.0)
    combined = round(sum(parts) / len(parts), 3) if parts else None

    return {
        "score": combined,
        "title_score": title_score,
        "venue_score": venue_score,
        "author_overlap_fraction": author_fraction,
        "year_match": year_match,
        "author_overlap": {
            "overlap": overlap,
            "provided": provided_count,
            "candidate": candidate_count,
        },
    }


def build_field_checks(citation: Dict[str, Any], work: Dict[str, Any]) -> Dict[str, Any]:
    input_title = citation.get("title")
    input_venue = citation.get("venue")
    input_year = parse_year(citation.get("year"))
    input_doi = clean_doi(citation.get("doi"))

    matched_title = work.get("title")
    matched_venue = work.get("venue")
    matched_year = work.get("year")
    matched_doi = clean_doi(work.get("doi"))

    input_authors = parse_provided_authors(citation.get("authors"))
    matched_authors = surnames_from_list(work.get("authors") or [])

    metrics = compute_match_metrics(
        citation,
        matched_title,
        matched_venue,
        work.get("authors") or [],
        matched_year,
    )

    doi_match = None
    if input_doi or matched_doi:
        doi_match = input_doi == matched_doi if input_doi and matched_doi else False

    return {
        "title": {
            "score": metrics.get("title_score"),
            "input": input_title,
            "matched": matched_title,
        },
        "authors": {
            "score": metrics.get("author_overlap_fraction"),
            "input": input_authors,
            "matched": matched_authors,
        },
        "year": {
            "match": metrics.get("year_match"),
            "input": input_year,
            "matched": matched_year,
        },
        "venue": {
            "score": metrics.get("venue_score"),
            "input": input_venue,
            "matched": matched_venue,
        },
        "doi": {
            "match": doi_match,
            "input": input_doi,
            "matched": matched_doi,
        },
        "score": metrics.get("score"),
    }


def select_best_overall(
    best_doi: Optional[Dict[str, Any]],
    best_title: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    def best_score(entry: Optional[Dict[str, Any]]) -> float:
        if not entry:
            return -1.0
        score = entry.get("score")
        return score if score is not None else -1.0

    if not best_doi and not best_title:
        return None
    doi_score = best_score(best_doi)
    title_score = best_score(best_title)
    if doi_score >= title_score:
        candidate = best_doi
        kind = "doi"
    else:
        candidate = best_title
        kind = "title"
    if not candidate:
        return None
    return {
        "kind": kind,
        "provider": candidate.get("provider"),
        "score": candidate.get("score"),
        "field_checks": candidate.get("field_checks"),
    }


def none_as_string(value: Optional[Any]) -> Any:
    return "none" if value is None else value


def build_results_grouped(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results_grouped: Dict[str, Dict[str, Any]] = {
        "doi_lookup": {},
        "title_search": {},
    }
    for kind_key in ("doi_lookup", "title_search"):
        provider_entries = evidence.get(kind_key) or {}
        for provider_name, entry in provider_entries.items():
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            item.pop("provider", None)
            if kind_key == "doi_lookup":
                meta = item.get("meta")
                if isinstance(meta, dict):
                    meta.pop("settings", None)
            if kind_key == "doi_lookup":
                data = item.get("data")
                if isinstance(data, dict) and "results" not in data:
                    item["data"] = {"results": [data]}
            data_results: List[Dict[str, Any]] = []
            data = item.get("data")
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                data_results = data.get("results") or []
            results_grouped[kind_key][provider_name] = {
                "results": data_results,
                "request_data": {
                    "status": item.get("status"),
                    "meta": item.get("meta"),
                },
            }
    return results_grouped


def build_grouped_row(
    citation: Dict[str, Any],
    verdict: Dict[str, Any],
    best_overall: Optional[Dict[str, Any]],
    lookup: Dict[str, Any],
    result: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    grouped_row: Dict[str, Any] = {
        "input": dict(citation),
        "verdict": {
            "doi": verdict.get("doi"),
            "title": verdict.get("title"),
            "best_match_kind": none_as_string(best_overall.get("kind") if best_overall else None),
            "best_match_provider": none_as_string(best_overall.get("provider") if best_overall else None),
            "best_match_score": none_as_string(best_overall.get("score") if best_overall else None),
        },
        "field_comparison": best_overall.get("field_checks") if best_overall else None,
        "results": build_results_grouped(evidence),
    }
    if lookup.get("doi_normalized"):
        grouped_row["input"]["doi_normalized"] = lookup.get("doi_normalized")
    if result.get("ignored"):
        grouped_row["ignored"] = True
    if result.get("reason"):
        grouped_row["reason"] = result["reason"]
    return grouped_row


def slim_best(best_entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not best_entry or not best_entry.get("best"):
        return None
    candidate = best_entry["best"]
    return {
        "provider": candidate.get("provider"),
        "rank": candidate.get("rank"),
        "score": candidate.get("score"),
    }


def aggregate_status(entries: Dict[str, Any]) -> str:
    if not isinstance(entries, dict) or not entries:
        return "unknown"
    found_providers = [
        provider
        for provider, entry in entries.items()
        if isinstance(entry, dict) and entry.get("status") == "found"
    ]
    if len(found_providers) >= 2:
        return "both"
    if len(found_providers) == 1:
        return found_providers[0]
    statuses = [
        entry.get("status")
        for entry in entries.values()
        if isinstance(entry, dict) and entry.get("status") is not None
    ]
    if not statuses:
        return "unknown"
    if "error" in statuses:
        return "error"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    if "not_found" in statuses:
        return "none"
    return "unknown"

def extract_year_from_crossref(message: Dict[str, Any]) -> Optional[int]:
    for key in ("issued", "published-print", "published-online", "created"):
        section = message.get(key)
        if not section:
            continue
        date_parts = section.get("date-parts") or section.get("date_parts") or []
        if date_parts and date_parts[0]:
            year = parse_year(date_parts[0][0])
            if year:
                return year
    return None


DOI_EXTRACT_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
DOI_LEADING_RE = re.compile(r"^(?:urn:)?doi\s*[:/]\s*", re.IGNORECASE)
DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi.org/",
    "dx.doi.org/",
)
DOI_TRAILING_PUNCT = ").,;:]>}\"'"


def clean_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # Remove invisible junk that can sneak into copied DOIs.
    raw = raw.replace("\u200b", "").replace("\ufeff", "").strip()
    raw = raw.strip("<>\"'")
    raw = raw.lower()
    raw = DOI_LEADING_RE.sub("", raw).strip()
    for prefix in DOI_URL_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = raw.strip()
    match = DOI_EXTRACT_RE.search(raw)
    if match:
        raw = match.group(0)
    raw = raw.rstrip(DOI_TRAILING_PUNCT).strip()
    return raw or None


def is_excluded_citation(citation: Dict[str, Any]) -> Optional[str]:
    doi = clean_doi(citation.get("doi"))
    if doi and doi in EXCLUDED_DOIS:
        return f"doi:{doi}"
    if doi and "/" in doi:
        suffix = doi.split("/", 1)[1]
        if suffix in EXCLUDED_DOI_SUFFIXES:
            return f"doi_suffix:{suffix}"
    title = normalize_text(citation.get("title"))
    if title:
        for phrase in EXCLUDED_TITLE_PHRASES:
            if phrase in title:
                return f"title_contains:{phrase}"
    return None


def format_author_name(author: Dict[str, Any]) -> Optional[str]:
    family = author.get("family")
    given = author.get("given")
    if family and given:
        return f"{family}, {given}"
    return author.get("name") or family or given


def summarize_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    authors: List[str] = []
    for author in item.get("author", []) or []:
        name = format_author_name(author)
        if name:
            authors.append(name)
    summary = {
        "doi": item.get("DOI"),
        "title": (item.get("title") or [None])[0],
        "year": extract_year_from_crossref(item),
        "venue": (item.get("container-title") or [None])[0],
        "authors": authors,
        "type": item.get("type"),
        "publisher": item.get("publisher"),
        "url": item.get("URL"),
        "is_referenced_by_count": item.get("is-referenced-by-count"),
    }
    score = item.get("score")
    if score is not None:
        summary["crossref_score"] = score
    return summary


def summarize_openalex_item(work: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(work, dict):
        return {}
    host_venue = work.get("host_venue") or {}
    authors: List[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    primary_location = work.get("primary_location") or {}
    summary = {
        "id": work.get("id"),
        "doi": clean_doi(work.get("doi")),
        "title": work.get("display_name") or work.get("title"),
        "year": work.get("publication_year"),
        "venue": host_venue.get("display_name"),
        "publisher": host_venue.get("publisher"),
        "authors": authors,
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "url": primary_location.get("landing_page_url"),
    }
    score = work.get("relevance_score") or work.get("score")
    if score is not None:
        summary["openalex_score"] = score
    return summary


def summarize_search_results(
    items: Iterable[Dict[str, Any]],
    citation: Dict[str, Any],
    max_results: int,
    min_title_score: float,
    summarize_item,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    results = [summarize_item(item) for item in items]
    for item in results:
        item["match"] = compute_match_metrics(
            citation,
            item.get("title"),
            item.get("venue"),
            item.get("authors") or [],
            item.get("year"),
        )
    if min_title_score is not None:
        results = [
            item
            for item in results
            if (item.get("match", {}).get("title_score") or 0.0) >= min_title_score
        ]
    results.sort(
        key=lambda entry: (
            entry.get("match", {}).get("score") is not None,
            entry.get("match", {}).get("score") or 0,
        ),
        reverse=True,
    )
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx
    summary = dict(meta or {})
    summary.update(
        {
            "max_results": max_results,
            "min_title_score": min_title_score,
            "results": results[:max_results],
        }
    )
    return summary


def summarize_crossref_search(
    raw: Optional[Dict[str, Any]],
    citation: Dict[str, Any],
    max_results: int,
    min_title_score: float,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    message = raw.get("message")
    if not isinstance(message, dict):
        return None
    items = message.get("items") or []
    meta = {
        "total_results": message.get("total-results"),
        "items_per_page": message.get("items-per-page"),
    }
    return summarize_search_results(
        items,
        citation,
        max_results,
        min_title_score,
        summarize_crossref_item,
        meta,
    )


def summarize_openalex_search(
    raw: Optional[Dict[str, Any]],
    citation: Dict[str, Any],
    max_results: int,
    min_title_score: float,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    meta = raw.get("meta") or {}
    items = raw.get("results") or []
    meta_summary = {
        "count": meta.get("count"),
        "page": meta.get("page"),
        "per_page": meta.get("per_page"),
    }
    return summarize_search_results(
        items,
        citation,
        max_results,
        min_title_score,
        summarize_openalex_item,
        meta_summary,
    )


def summarize_crossref_work(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    message = raw.get("message")
    if not isinstance(message, dict):
        return None
    return summarize_crossref_item(message)


def summarize_openalex_work(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return summarize_openalex_item(raw)


class CrossrefClient:
    def __init__(self, session: requests.Session):
        self.session = session
        self.cache: Dict[str, Any] = {}

    def lookup(self, doi: str) -> Dict[str, Any]:
        doi = clean_doi(doi)
        if not doi:
            return {"status": "missing", "raw": None, "http_status": None}
        if doi in self.cache:
            return self.cache[doi]
        url = f"https://api.crossref.org/works/{quote(doi)}"
        request_url = url
        res = get_with_retries(self.session, url)
        resp = res.get("response")
        if res["status"] == "error":
            result = build_request_error(res.get("error"), request_url)
            self.cache[doi] = result
            return result
        result = build_response_result(resp, request_url, not_found_statuses=(404,))
        self.cache[doi] = result
        return result

    def search_title(self, title: str, rows: int = 5) -> Dict[str, Any]:
        clean_title = re.sub(r"\s+", " ", title).strip()
        params = {"query.bibliographic": clean_title, "rows": rows}
        url = "https://api.crossref.org/works"
        request_url = build_request_url(url, params=params)
        res = get_with_retries(self.session, url, params=params)
        resp = res.get("response")
        if res["status"] == "error":
            return build_request_error(res.get("error"), request_url)
        return build_response_result(resp, request_url)


class OpenAlexClient:
    def __init__(self, session: requests.Session):
        self.session = session
        self.search_cache: Dict[str, Any] = {}
        self.doi_cache: Dict[str, Any] = {}

    def search_by_title(self, title: str, per_page: int = 5) -> Dict[str, Any]:
        clean_title = re.sub(r"\s+", " ", title).strip()
        title_key = clean_title.lower()
        if title_key in self.search_cache:
            return self.search_cache[title_key]
        params = {"filter": f"title.search:{clean_title}", "per-page": per_page}
        primary_url = build_request_url("https://api.openalex.org/works", params=params)
        res = get_with_retries(self.session, "https://api.openalex.org/works", params=params)
        resp = res.get("response")
        extra = {
            "request_url_primary": primary_url,
            "request_url_used": "primary",
        }
        if res["status"] == "error":
            result = build_request_error(res.get("error"), primary_url, extra=extra)
            self.search_cache[title_key] = result
            return result
        result = build_response_result(resp, primary_url, extra=extra)
        self.search_cache[title_key] = result
        return result

    def lookup_by_doi(self, doi: str) -> Dict[str, Any]:
        doi = clean_doi(doi)
        if not doi:
            return {"status": "missing", "raw": None, "http_status": None}
        if doi in self.doi_cache:
            return self.doi_cache[doi]
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        request_url = url
        res = get_with_retries(self.session, url)
        resp = res.get("response")
        if res["status"] == "error":
            result = build_request_error(res.get("error"), request_url)
            self.doi_cache[doi] = result
            return result
        result = build_response_result(resp, request_url, not_found_statuses=(404,))
        self.doi_cache[doi] = result
        return result


class CitationValidator:
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        max_results: int = 5,
        min_title_score: float = MIN_TITLE_SCORE,
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.crossref = CrossrefClient(self.session)
        self.openalex = OpenAlexClient(self.session)
        self.max_results = max_results
        self.min_title_score = min_title_score

    def validate(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        excluded_reason = is_excluded_citation(citation)
        doi_normalized = clean_doi(citation.get("doi"))
        if excluded_reason:
            return {
                "ignored": True,
                "reason": excluded_reason,
                "verdict": {"doi": "ignored", "title": "ignored"},
                "best_match": {"doi": None, "title": None},
                "field_checks": {"doi": None, "title": None},
                "lookup": {
                    "doi_normalized": doi_normalized,
                    "queries": [],
                },
                "results": {},
            }

        doi = doi_normalized
        title = citation.get("title")
        result: Dict[str, Any] = {
            "ignored": False,
            "lookup": {
                "doi_normalized": doi,
                "queries": [],
            },
        }

        def build_result_entry(
            payload: Dict[str, Any],
            summary_func,
            provider: str,
            kind: str,
        ) -> Dict[str, Any]:
            http_status = payload.get("http_status")
            request_url = payload.get("request_url")
            entry = {
                "status": payload.get("status"),
                "provider": provider,
                "meta": {
                    "http_status": http_status,
                    "url": request_url,
                },
            }
            if payload.get("status") == "found":
                summary = summary_func(payload.get("raw"))
                if summary is not None:
                    if kind == "doi":
                        summary["match"] = compute_match_metrics(
                            citation,
                            summary.get("title"),
                            summary.get("venue"),
                            summary.get("authors") or [],
                            summary.get("year"),
                        )
                        entry["data"] = summary
                        ref_count = summary.get("is_referenced_by_count")
                        if ref_count is None:
                            ref_count = summary.get("cited_by_count")
                        if ref_count is not None:
                            entry["meta"]["ref_count"] = ref_count
                    else:
                        results = None
                        if isinstance(summary, dict):
                            settings_updates = {
                                k: summary.get(k) for k in SEARCH_SETTINGS_KEYS if k in summary
                            }
                            response_updates = {
                                k: summary.get(k) for k in SEARCH_RESPONSE_KEYS if k in summary
                            }
                            if settings_updates:
                                entry["meta"].setdefault("settings", {}).update(settings_updates)
                            if response_updates or http_status is not None or request_url:
                                response_bucket = entry["meta"].setdefault("response", {})
                                response_bucket.update(response_updates)
                                if http_status is not None:
                                    response_bucket["http_status"] = http_status
                                if request_url:
                                    response_bucket["url"] = request_url
                            data_payload = dict(summary)
                            for key in SEARCH_SETTINGS_KEYS | SEARCH_RESPONSE_KEYS:
                                data_payload.pop(key, None)
                            entry["data"] = data_payload
                            results = data_payload.get("results")
                        else:
                            entry["data"] = summary
                            results = None
                        if isinstance(results, list) and len(results) == 0:
                            entry["status"] = "not_found"
            elif payload.get("status") == "error":
                entry["error"] = payload.get("error")
            if request_url:
                result["lookup"]["queries"].append(
                    {
                        "provider": provider,
                        "kind": kind,
                        "url": request_url,
                        "status": http_status,
                    }
                )
            return entry

        if doi:
            cr_lookup = self.crossref.lookup(doi)
            oa_lookup = self.openalex.lookup_by_doi(doi)
            result["doi_lookup"] = {
                "crossref": build_result_entry(cr_lookup, summarize_crossref_work, "crossref", "doi"),
                "openalex": build_result_entry(oa_lookup, summarize_openalex_work, "openalex", "doi"),
            }
            # DOI agency lookup removed (was noisy and unused).
        else:
            result["doi_lookup"] = {
                "crossref": {"status": "skipped", "reason": "missing_doi"},
                "openalex": {"status": "skipped", "reason": "missing_doi"},
            }

        if title:
            oa_search = self.openalex.search_by_title(title, per_page=self.max_results)
            cr_search = self.crossref.search_title(title, rows=self.max_results)
            result["title_search"] = {
                "openalex": build_result_entry(
                    oa_search,
                    lambda raw: summarize_openalex_search(
                        raw, citation, self.max_results, self.min_title_score
                    ),
                    "openalex",
                    "title",
                ),
                "crossref": build_result_entry(
                    cr_search,
                    lambda raw: summarize_crossref_search(
                        raw, citation, self.max_results, self.min_title_score
                    ),
                    "crossref",
                    "title",
                ),
            }
        else:
            result["title_search"] = {
                "openalex": {"status": "skipped", "reason": "missing_title"},
                "crossref": {"status": "skipped", "reason": "missing_title"},
            }

        best_matches_full = self.select_best_matches(citation, result)
        field_checks = {"doi": None, "title": None}
        doi_best = best_matches_full.get("doi") if isinstance(best_matches_full, dict) else None
        if doi_best and doi_best.get("best") and doi_best["best"].get("work"):
            field_checks["doi"] = build_field_checks(citation, doi_best["best"]["work"])
        title_best = best_matches_full.get("title") if isinstance(best_matches_full, dict) else None
        if title_best and title_best.get("best") and title_best["best"].get("work"):
            field_checks["title"] = build_field_checks(citation, title_best["best"]["work"])

        best_matches = {
            "doi": slim_best(doi_best),
            "title": slim_best(title_best),
        }

        verdict = {
            "doi": aggregate_status(result.get("doi_lookup", {})),
            "title": aggregate_status(result.get("title_search", {})),
        }

        return {
            "ignored": result["ignored"],
            "verdict": verdict,
            "lookup": result["lookup"],
            "results": {
                "doi_lookup": result["doi_lookup"],
                "title_search": result["title_search"],
            },
            "best_match": best_matches,
            "field_checks": field_checks,
        }

    def select_best_matches(
        self,
        citation: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        def build_candidate_from_summary(
            provider: str,
            kind: str,
            summary: Dict[str, Any],
            match: Dict[str, Any],
            rank: Optional[int] = None,
        ) -> Dict[str, Any]:
            candidate = {
                "provider": provider,
                "kind": kind,
                "score": match.get("score"),
                "match": match,
                "work": {
                    "doi": summary.get("doi"),
                    "title": summary.get("title"),
                    "year": summary.get("year"),
                    "venue": summary.get("venue"),
                    "authors": summary.get("authors"),
                    "type": summary.get("type"),
                    "publisher": summary.get("publisher"),
                    "url": summary.get("url"),
                },
            }
            if rank is not None:
                candidate["rank"] = rank
            return candidate

        def candidate_score(candidate: Optional[Dict[str, Any]]) -> float:
            if not candidate:
                return -1.0
            score = candidate.get("score")
            return score if score is not None else -1.0

        def build_best_for_kind(kind: str) -> Optional[Dict[str, Any]]:
            candidates: Dict[str, Optional[Dict[str, Any]]] = {"crossref": None, "openalex": None}
            if kind == "doi":
                doi_lookup = result.get("doi_lookup", {})
                for provider in ("crossref", "openalex"):
                    entry = doi_lookup.get(provider) or {}
                    if entry.get("status") != "found":
                        continue
                    summary = entry.get("data") or {}
                    if not summary:
                        continue
                    match = summary.get("match") or {}
                    candidates[provider] = build_candidate_from_summary(provider, "doi", summary, match)
            elif kind == "title":
                title_search = result.get("title_search", {})
                for provider in ("openalex", "crossref"):
                    entry = title_search.get(provider) or {}
                    summary = entry.get("data") or {}
                    results = summary.get("results") or []
                    if not results:
                        continue
                    top = results[0]
                    match = top.get("match") or {}
                    candidates[provider] = build_candidate_from_summary(
                        provider,
                        "title",
                        top,
                        match,
                        rank=top.get("rank"),
                    )

            if not any(candidates.values()):
                return None

            oa_score = candidate_score(candidates.get("openalex"))
            cr_score = candidate_score(candidates.get("crossref"))
            if cr_score > oa_score:
                best_provider = "crossref"
            elif oa_score > cr_score:
                best_provider = "openalex"
            else:
                best_provider = "crossref" if candidates.get("crossref") else "openalex"

            return {
                "kind": kind,
                "best_provider": best_provider,
                "best": candidates.get(best_provider),
                "providers": {
                    "openalex": candidates.get("openalex"),
                    "crossref": candidates.get("crossref"),
                },
                "comparison": {
                    "openalex_score": oa_score if oa_score >= 0 else None,
                    "crossref_score": cr_score if cr_score >= 0 else None,
                    "better": best_provider,
                },
            }

        return {
            "doi": build_best_for_kind("doi"),
            "title": build_best_for_kind("title"),
        }


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def process_file(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    delay: float = 0.0,
    max_results: int = 5,
    min_title_score: float = MIN_TITLE_SCORE,
    progress_bar: bool = True,
) -> None:
    validator = CitationValidator(
        max_results=max_results,
        min_title_score=min_title_score,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_records = 0
    processed_citations = 0
    with output_path.open("w") as handle:
        handle.write("[\n")
        first = True
        iterator = load_jsonl(input_path)
        if progress_bar:
            from tqdm import tqdm  # type: ignore

            total = count_jsonl(input_path)
            if limit is not None:
                total = min(total, limit)
            iterator = tqdm(iterator, total=total, desc="records")

        for idx, record in enumerate(iterator):
            if limit is not None and idx >= limit:
                break
            citations = record.get("citations_parsed") or []
            record_id = record.get("id")
            record_field = record.get("field")
            record_entity = record.get("entity")
            grouped_results: Dict[str, Any] = {}
            for c_idx, citation in enumerate(citations):
                result = validator.validate(citation)
                citation_uid = f"{record_id}::c{c_idx}" if record_id is not None else f"c{c_idx}"
                best_match = result.get("best_match") or {}
                field_checks = result.get("field_checks") or {}
                evidence = result.get("results") or {}
                verdict = result.get("verdict") or {}
                lookup = result.get("lookup") or {}

                def merge_best(
                    best_entry: Optional[Dict[str, Any]],
                    checks: Optional[Dict[str, Any]],
                ) -> Optional[Dict[str, Any]]:
                    if not best_entry:
                        return None
                    merged = dict(best_entry)
                    if checks is not None:
                        merged["field_checks"] = checks
                    return merged

                best_doi = merge_best(best_match.get("doi"), field_checks.get("doi"))
                best_title = merge_best(best_match.get("title"), field_checks.get("title"))

                best_overall = select_best_overall(best_doi, best_title)

                grouped_results[citation_uid] = build_grouped_row(
                    citation,
                    verdict,
                    best_overall,
                    lookup,
                    result,
                    evidence,
                )

                processed_citations += 1
                if delay:
                    time.sleep(delay)

            output_record = {
                "id": record_id,
                "field": record_field,
                "entity": record_entity,
                "citations_text": record.get("citations_text"),
                "citations_parsed": record.get("citations_parsed"),
                "citations": grouped_results,
            }
            if not first:
                handle.write(",\n")
            handle.write(json.dumps(output_record, indent=2))
            first = False
            processed_records += 1
        handle.write("\n]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LLM-generated citations against Crossref and OpenAlex.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Llama-2-7b-chat-hf_citations (2).jsonl"),
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation_results.json"),
        help="Path for output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for sample run.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between requests.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max number of title-search results per source to store.",
    )
    parser.add_argument(
        "--min-title-score",
        type=float,
        default=MIN_TITLE_SCORE,
        help="Minimum title similarity score required for title search results.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_file(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        delay=args.delay,
        max_results=args.max_results,
        min_title_score=args.min_title_score,
        progress_bar=True,
    )


if __name__ == "__main__":
    main()
