"""
HypothesisForge - Document Ingestion & Passage Retrieval
Extracts page-indexed text from PDFs using pypdf and chunks passages for evidence retrieval.
"""

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import List, Tuple
import pypdf


def make_stable_paper_id(title: str, identifier: str = "") -> str:
    """Generates a deterministic, stable Paper ID from title/identifier so IDs never shift between runs."""
    basis = re.sub(r"[^a-z0-9]", "", title.lower()) or identifier.lower()
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:6].upper()
    return f"P-{digest}"


@dataclass
class TextChunk:
    chunk_id: int
    paper_id: str
    title: str
    page: int
    text: str


@dataclass
class PaperDocument:
    paper_id: str
    title: str
    year: str
    pages: List[Tuple[int, str]]
    formatted_text: str
    extraction_status: str = "ok"
    relevance_score: float = 1.0
    chunks: List[TextChunk] = field(default_factory=list)


def format_pages_for_prompt(pages: List[Tuple[int, str]]) -> str:
    """Formats extracted pages as [Page n]: blocks expected by Prompt 01."""
    blocks = []
    for page_num, text in pages:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            blocks.append(f"[Page {page_num}]:\n{cleaned}")
    return "\n\n".join(blocks)


def chunk_pages(
    paper_id: str,
    title: str,
    pages: List[Tuple[int, str]],
    max_words: int = 110,
    start_id: int = 1,
) -> List[TextChunk]:
    """Splits page text into sentence-aware passages with page metadata."""
    chunks: List[TextChunk] = []
    idx = start_id
    for page_num, text in pages:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        current_words: List[str] = []
        for sent in sentences:
            sent_words = sent.split()
            if current_words and len(current_words) + len(sent_words) > max_words:
                chunks.append(
                    TextChunk(
                        chunk_id=idx,
                        paper_id=paper_id,
                        title=title,
                        page=page_num,
                        text=" ".join(current_words),
                    )
                )
                idx += 1
                current_words = []
            current_words.extend(sent_words)
        if current_words:
            chunks.append(
                TextChunk(
                    chunk_id=idx,
                    paper_id=paper_id,
                    title=title,
                    page=page_num,
                    text=" ".join(current_words),
                )
            )
            idx += 1
    return chunks


def extract_pdf_bytes(file_bytes: bytes, filename: str, paper_id: str = "") -> PaperDocument:
    """Extracts text page-by-page from an uploaded PDF file."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        pages.append((i, txt))

    formatted = format_pages_for_prompt(pages)
    status = "ok" if formatted.strip() else "empty_or_scanned"

    # Infer title from metadata or filename
    meta_title = ""
    if reader.metadata and getattr(reader.metadata, "title", None):
        meta_title = str(reader.metadata.title).strip()
    title = meta_title or filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")

    if not paper_id:
        paper_id = make_stable_paper_id(title, filename)

    # Infer year if present in first page text
    year = "not stated"
    if pages:
        yr_match = re.search(r"\b(201\d|202\d)\b", pages[0][1])
        if yr_match:
            year = yr_match.group(1)

    chunks = chunk_pages(paper_id=paper_id, title=title, pages=pages)
    return PaperDocument(
        paper_id=paper_id,
        title=title,
        year=year,
        pages=pages,
        formatted_text=formatted,
        extraction_status=status,
        chunks=chunks,
    )


def _is_boilerplate_chunk(text: str) -> bool:
    """Detects back-matter boilerplate (Acknowledgements, Competing Interests, Author Contributions)."""
    low = text.lower()
    markers = (
        "competing interests",
        "author contributions",
        "publisher's note",
        "acknowledgements",
        "acknowledgments",
        "correspondence and requests for materials",
        "supplementary information",
    )
    return any(m in low for m in markers)


def retrieve_passages(question: str, all_chunks: List[TextChunk], top_k: int = 5) -> Tuple[str, List[TextChunk]]:
    """
    Retrieves top-k relevant passages using TF-IDF cosine similarity + exact acronym/stem boost,
    while filtering out back-matter boilerplate (Acknowledgements, Competing Interests).
    Returns empty string if no passages have meaningful relevance.
    """
    if not all_chunks:
        return "", []

    q_tokens = set(re.findall(r"[a-zA-Z0-9\-]+", question.lower()))
    stop_words = {
        "what", "which", "does", "that", "this", "with", "from", "have", "about",
        "they", "their", "were", "when", "where", "how", "can", "could", "would",
        "should", "build", "make", "using", "used", "want", "need", "give", "tell",
        "show", "explain", "some", "more", "most", "into", "than", "then", "also",
        "paper", "papers", "study", "studies", "work", "any", "the", "and", "for",
    }
    q_terms = {t for t in q_tokens if len(t) > 2 and t not in stop_words}
    if not q_terms:
        return "", []

    # Filter out back-matter boilerplate unless user specifically asks about authors/funding
    wants_meta = any(k in question.lower() for k in ("author", "funding", "acknowledg", "grant", "conflict of interest"))
    candidate_chunks = all_chunks if wants_meta else [c for c in all_chunks if not _is_boilerplate_chunk(c.text)]
    if not candidate_chunks:
        candidate_chunks = all_chunks

    # Compute TF-IDF cosine similarity across candidate chunks
    tfidf_scores: List[float] = [0.0] * len(candidate_chunks)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
        mat = vec.fit_transform([c.text for c in candidate_chunks])
        q_vec = vec.transform([question])
        sims = cosine_similarity(q_vec, mat).flatten()
        tfidf_scores = [float(s) for s in sims]
    except Exception:
        pass

    scored: List[Tuple[float, TextChunk]] = []
    acronyms = re.findall(r"\b[A-Z0-9\-]{2,}\b", question)
    for idx_c, chunk in enumerate(candidate_chunks):
        c_text_lower = chunk.text.lower()
        # Stem/prefix-aware lexical score (handles plurals, hyphens, e.g., dataset/datasets, pre-processing/preprocessing)
        c_norm = re.sub(r"[-_]", "", c_text_lower)
        lex_score = 0.0
        for t in q_terms:
            t_norm = re.sub(r"[-_]", "", t)
            stem = t_norm[: max(4, len(t_norm) - 2)] if len(t_norm) >= 5 else t_norm
            if re.search(r"\b" + re.escape(t) + r"\b", c_text_lower):
                lex_score += 2.0
            elif stem in c_norm:
                lex_score += 1.2

        # Exact uppercase acronyms / numbers boost (e.g., PACS, ERM, ViT, AUC)
        for token in acronyms:
            if token in chunk.text:
                lex_score += 3.0

        total_score = (tfidf_scores[idx_c] * 10.0) + lex_score
        if total_score > 0.35:
            scored.append((total_score, chunk))

    if not scored:
        return "", []

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in scored[:top_k]]

    formatted_lines = []
    for idx, chunk in enumerate(selected, start=1):
        formatted_lines.append(f"[{idx}] ({chunk.paper_id}, Page {chunk.page}): {chunk.text}")

    return "\n".join(formatted_lines), selected
