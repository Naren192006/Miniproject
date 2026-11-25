"""
Unit tests for free scholarly paper discovery (arXiv + OpenAlex) and PDF/abstract ingestion.
"""

from unittest.mock import patch, MagicMock
from hypothesisforge.paper_search import (
    DiscoveredPaper,
    parse_arxiv_feed,
    parse_openalex_response,
    discover_papers,
    fetch_paper_documents,
)


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-05T00:00:00Z</published>
    <title>Vision Transformers for Domain Generalization</title>
    <summary>We study Vision Transformer generalization on PACS and OfficeHome achieving 86.2% Accuracy.</summary>
    <author><name>Alice Researcher</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

SAMPLE_OPENALEX_JSON = {
    "results": [
        {
            "id": "https://openalex.org/W123456",
            "title": "Vision Transformers for Domain Generalization",
            "publication_year": 2024,
            "abstract_inverted_index": {"We": [0], "evaluate": [1], "ViT": [2]},
            "authorships": [{"author": {"display_name": "Alice Researcher"}}],
            "open_access": {"oa_url": None},
            "primary_location": {"landing_page_url": "https://example.org/paper1"},
        },
        {
            "id": "https://openalex.org/W789012",
            "title": "Empirical Risk Minimization Under Covariate Shift",
            "publication_year": 2023,
            "abstract_inverted_index": {"ERM": [0], "achieves": [1], "80.2%": [2], "on": [3], "PACS.": [4]},
            "authorships": [{"author": {"display_name": "Bob Scientist"}}],
            "open_access": {"oa_url": None},
            "primary_location": {"landing_page_url": "https://example.org/paper2"},
        },
    ]
}


def test_parse_arxiv_feed():
    papers = parse_arxiv_feed(SAMPLE_ARXIV_XML)
    assert len(papers) == 1
    assert papers[0].title == "Vision Transformers for Domain Generalization"
    assert papers[0].year == "2024"
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.00001v1"
    assert papers[0].source == "arXiv"


def test_parse_openalex_response():
    papers = parse_openalex_response(SAMPLE_OPENALEX_JSON)
    assert len(papers) == 2
    assert papers[1].title == "Empirical Risk Minimization Under Covariate Shift"
    assert papers[1].abstract == "ERM achieves 80.2% on PACS."
    assert papers[1].year == "2023"


def test_discover_papers_deduplication():
    with patch("hypothesisforge.paper_search.search_arxiv", return_value=parse_arxiv_feed(SAMPLE_ARXIV_XML)), \
         patch("hypothesisforge.paper_search.search_openalex", return_value=parse_openalex_response(SAMPLE_OPENALEX_JSON)):
        results = discover_papers("Domain Generalization", max_papers=5, min_relevance=0.0)
        assert len(results) == 2
        # Duplicate title keeps the arXiv entry which has pdf_url
        vit_paper = next(p for p in results if "Vision Transformers" in p.title)
        assert vit_paper.pdf_url == "http://arxiv.org/pdf/2401.00001v1"


def test_discover_papers_ieee_and_30_limit():
    ieee_mock = parse_openalex_response(SAMPLE_OPENALEX_JSON)
    for p in ieee_mock:
        p.source = "IEEE"
    with patch("hypothesisforge.paper_search.search_ieee", return_value=ieee_mock):
        results = discover_papers("Domain Generalization", max_papers=30, source_mode="IEEE only", min_relevance=0.0)
        assert len(results) == 2
        assert all(p.source == "IEEE" for p in results)


def test_fetch_paper_documents_abstract_fallback():
    discovered = [
        DiscoveredPaper(
            title="Test Abstract Paper",
            year="2024",
            abstract="ResNet-50 achieves 82.0% Accuracy on PACS.",
            source="OpenAlex",
            pdf_url=None,
        )
    ]
    docs = fetch_paper_documents(discovered, download_pdfs=True)
    assert len(docs) == 1
    assert docs[0].paper_id.startswith("P-")
    assert docs[0].extraction_status == "abstract_only"
    assert "ResNet-50 achieves 82.0%" in docs[0].formatted_text
    assert len(docs[0].chunks) >= 1
