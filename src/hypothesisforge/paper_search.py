"""
HypothesisForge - Free Scholarly Paper Discovery (arXiv + OpenAlex)
Searches arXiv and OpenAlex without requiring any API keys, downloads accessible PDFs,
and converts results into PaperDocument objects for the analysis pipeline.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import requests

from .ingestion import (
    PaperDocument,
    chunk_pages,
    extract_pdf_bytes,
    format_pages_for_prompt,
    make_stable_paper_id,
)


ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_API_URL = "https://api.openalex.org/works"
USER_AGENT = "HypothesisForge/0.1 (Research Gap Discovery Prototype)"


@dataclass
class DiscoveredPaper:
    title: str
    year: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    source: str = "arXiv"
    url: str = ""
    pdf_url: Optional[str] = None


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstructs plain abstract text from OpenAlex abstract_inverted_index."""
    if not inverted_index:
        return ""
    max_pos = max((pos for positions in inverted_index.values() for pos in positions), default=-1)
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return _clean_text(" ".join(w for w in words if w))


def parse_arxiv_feed(xml_text: str) -> List[DiscoveredPaper]:
    """Parses arXiv Atom XML response into DiscoveredPaper objects."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    papers: List[DiscoveredPaper] = []

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)

        title = _clean_text(title_el.text if title_el is not None and title_el.text else "")
        if not title:
            continue

        abstract = _clean_text(summary_el.text if summary_el is not None and summary_el.text else "")
        published = published_el.text if published_el is not None and published_el.text else ""
        year = published[:4] if len(published) >= 4 and published[:4].isdigit() else "not stated"
        abs_url = _clean_text(id_el.text if id_el is not None and id_el.text else "")

        authors = [
            _clean_text(a.find("atom:name", ns).text)
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None and a.find("atom:name", ns).text
        ]

        pdf_url: Optional[str] = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        if not pdf_url and "/abs/" in abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

        papers.append(
            DiscoveredPaper(
                title=title,
                year=year,
                authors=authors,
                abstract=abstract,
                source="arXiv",
                url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def search_arxiv(topic: str, max_results: int = 5, timeout: int = 15) -> List[DiscoveredPaper]:
    """Queries the free arXiv API for papers matching the topic."""
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(
        ARXIV_API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_arxiv_feed(resp.text)


def parse_openalex_response(data: Dict) -> List[DiscoveredPaper]:
    """Parses OpenAlex JSON works response into DiscoveredPaper objects."""
    papers: List[DiscoveredPaper] = []
    for work in data.get("results", []):
        title = _clean_text(work.get("title") or "")
        if not title:
            continue
        year = str(work.get("publication_year") or "not stated")
        abstract = _reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
        authors = [
            _clean_text(auth.get("author", {}).get("display_name", ""))
            for auth in work.get("authorships", [])
            if auth.get("author", {}).get("display_name")
        ]
        oa_info = work.get("open_access") or {}
        primary_loc = work.get("primary_location") or {}
        pdf_url = oa_info.get("oa_url") or primary_loc.get("pdf_url")
        url = work.get("id") or primary_loc.get("landing_page_url") or ""

        papers.append(
            DiscoveredPaper(
                title=title,
                year=year,
                authors=authors,
                abstract=abstract,
                source="OpenAlex",
                url=url,
                pdf_url=pdf_url,
            )
        )
    return papers


def search_openalex(topic: str, max_results: int = 5, timeout: int = 15) -> List[DiscoveredPaper]:
    """Queries the free OpenAlex API for scholarly works matching the topic."""
    params = {
        "search": topic,
        "per-page": min(max_results, 100),
        "filter": "has_abstract:true",
    }
    resp = requests.get(
        OPENALEX_API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_openalex_response(resp.json())


def search_ieee(topic: str, max_results: int = 5, timeout: int = 15) -> List[DiscoveredPaper]:
    """
    Queries IEEE-published papers (TPAMI, TKDE, CVPR, IEEE Access, etc.) via OpenAlex's
    IEEE publisher index (P4310319808) with no API key required.
    """
    params = {
        "search": topic,
        "per-page": min(max_results, 100),
        "filter": "primary_location.source.publisher_lineage:P4310319808,has_abstract:true",
    }
    resp = requests.get(
        OPENALEX_API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    papers = parse_openalex_response(resp.json())
    for p in papers:
        p.source = "IEEE"
    return papers


def score_topic_relevance(topic: str, title: str, abstract: str) -> float:
    """
    Computes a 0.0-1.0 relevance score between the user's topic query and a paper's title/abstract.
    Filters out off-topic papers (e.g. time-series forecasting when searching domain generalization).
    """
    stop_words = {
        "and", "for", "the", "with", "from", "under", "using", "into", "via", "based", "on", "in", "of", "to", "a", "an"
    }
    topic_terms = [
        t for t in re.findall(r"[a-z0-9\-]+", topic.lower())
        if len(t) > 2 and t not in stop_words
    ]
    if not topic_terms:
        return 1.0

    title_lower = title.lower()
    abs_lower = abstract.lower()
    combined = f"{title_lower} {abs_lower}"

    # Count matched terms, weighting title matches higher
    matched_terms = sum(1 for t in topic_terms if t in combined)
    title_hits = sum(1 for t in topic_terms if t in title_lower)

    base_ratio = matched_terms / len(topic_terms)
    bonus = 0.25 if title_hits >= 1 else 0.0

    # Penalize obvious off-topic domain mismatches unless explicitly requested in topic
    if "domain generalization" in topic.lower() and "time series" in title_lower and "time series" not in topic.lower():
        return max(0.0, base_ratio - 0.5)

    return min(1.0, base_ratio + bonus)


def discover_papers(
    topic: str,
    max_papers: int = 5,
    source_mode: str = "arXiv + OpenAlex",
    min_relevance: float = 0.45,
) -> List[DiscoveredPaper]:
    """
    Discovers papers from arXiv, OpenAlex, and/or IEEE (via OpenAlex publisher index),
    deduplicating by normalized title and filtering out off-topic papers.
    """
    candidates: List[DiscoveredPaper] = []
    errors: List[str] = []
    fetch_count = min(max(max_papers * 2, 10), 60)

    if "arXiv" in source_mode or "All" in source_mode:
        try:
            candidates.extend(search_arxiv(topic, max_results=fetch_count))
        except Exception as exc:
            errors.append(f"arXiv search failed: {exc}")

    if "OpenAlex" in source_mode or "All" in source_mode:
        try:
            candidates.extend(search_openalex(topic, max_results=fetch_count))
        except Exception as exc:
            errors.append(f"OpenAlex search failed: {exc}")

    if "IEEE" in source_mode or "All" in source_mode:
        try:
            candidates.extend(search_ieee(topic, max_results=fetch_count))
        except Exception as exc:
            errors.append(f"IEEE search failed: {exc}")

    if not candidates and errors:
        raise RuntimeError(" | ".join(errors))

    # Deduplicate by normalized title, preferring entries with pdf_url
    seen: Dict[str, DiscoveredPaper] = {}
    for p in candidates:
        norm_title = re.sub(r"[^a-z0-9]", "", p.title.lower())
        if not norm_title:
            continue
        if norm_title not in seen:
            seen[norm_title] = p
        elif not seen[norm_title].pdf_url and p.pdf_url:
            seen[norm_title] = p

    # Filter by topic relevance
    scored: List[tuple[float, DiscoveredPaper]] = []
    for p in seen.values():
        rel = score_topic_relevance(topic, p.title, p.abstract)
        if rel >= min_relevance:
            scored.append((rel, p))

    scored.sort(key=lambda x: (x[0], x[1].pdf_url is not None, bool(x[1].abstract)), reverse=True)
    return [p for _, p in scored[:max_papers]]


def fetch_paper_documents(
    discovered: List[DiscoveredPaper],
    download_pdfs: bool = True,
    pdf_timeout: int = 20,
) -> List[PaperDocument]:
    """
    Converts DiscoveredPaper items into PaperDocument objects with stable Paper IDs,
    downloading PDFs when available or falling back to abstract-only ingestion.
    """
    docs: List[PaperDocument] = []

    for item in discovered:
        paper_id = make_stable_paper_id(item.title, item.url)
        doc: Optional[PaperDocument] = None

        if download_pdfs and item.pdf_url:
            try:
                resp = requests.get(
                    item.pdf_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=pdf_timeout,
                )
                content_type = resp.headers.get("Content-Type", "").lower()
                if resp.status_code == 200 and (b"%PDF" in resp.content[:1024] or "pdf" in content_type):
                    extracted = extract_pdf_bytes(resp.content, f"{paper_id}.pdf", paper_id=paper_id)
                    if extracted.extraction_status == "ok":
                        extracted.title = item.title
                        extracted.year = item.year
                        for c in extracted.chunks:
                            c.title = item.title
                        doc = extracted
            except Exception:
                doc = None

        # Fallback to abstract-only document if PDF unavailable or failed
        if doc is None:
            abstract_body = item.abstract or "Abstract not available from scholarly metadata."
            pages = [(1, f"Title: {item.title}\nAbstract: {abstract_body}")]
            formatted = format_pages_for_prompt(pages)
            chunks = chunk_pages(paper_id=paper_id, title=item.title, pages=pages)
            doc = PaperDocument(
                paper_id=paper_id,
                title=item.title,
                year=item.year,
                pages=pages,
                formatted_text=formatted,
                extraction_status="abstract_only",
                chunks=chunks,
            )

        docs.append(doc)

    return docs
