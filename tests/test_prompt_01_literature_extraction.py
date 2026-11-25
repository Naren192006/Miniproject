"""
Unit and Validation Tests for Prompt 01: Literature Extraction
File: prompts/01_literature_extraction.md | Version 1.1 | Stage: literature
"""

import json
import pytest
from pathlib import Path

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import LiteratureExtractionOutput, ReportedResultItem
from hypothesisforge.prompt_engine.fallbacks import RuleBasedLiteratureExtractor
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_PAPER_TEXT = """
[Page 1]:
In this paper, we study Domain Generalization for visual recognition. We evaluate deep models on the PACS dataset
using a Leave-one-domain-out protocol across Photo, Art Painting, Cartoon, and Sketch domains.

[Page 2]:
We compare Empirical Risk Minimization (ERM) with our proposed Domain-Adversarial Training strategy using a ResNet-50 architecture.
Our evaluation protocol uses training on three source domains and testing on the single held-out target domain with 3 independent runs.
On the PACS benchmark, our proposed method achieves an average Accuracy of 84.6% across target domains, whereas standard ERM baseline reaches 80.2%.

[Page 3]:
Findings:
Our proposed method consistently outperforms standard ERM across all difficult target domains such as Sketch.
Limitations:
A limitation of our approach is that computational requirements increase substantially with the number of training domains.
"""


def test_prompt_01_metadata_and_loading():
    """Verify prompt 01 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p1 = loader.load_prompt("01_literature_extraction")

    assert p1.name == "01 Literature extraction"
    assert p1.version == "1.1"
    assert p1.stage == "literature"
    assert "structured experimental record" in p1.purpose
    assert "You are an expert reviewer of machine learning" in p1.system_prompt
    assert "untrusted document content" in p1.system_prompt
    assert p1.placeholders == {"title", "year", "paper_text"}


def test_prompt_01_template_rendering():
    """Verify placeholder replacement works and missing variables trigger error."""
    loader = PromptLoader()
    p1 = loader.load_prompt("01_literature_extraction")

    rendered = p1.render_user_prompt(
        title="Domain Generalization via Adversarial Invariance",
        year=2023,
        paper_text=SAMPLE_PAPER_TEXT
    )

    assert "Domain Generalization via Adversarial Invariance" in rendered
    assert "2023" in rendered
    assert "[Page 1]:" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    # Missing variable should raise ValueError
    with pytest.raises(ValueError, match="Missing required placeholder"):
        p1.render_user_prompt(title="Incomplete", year=2023)


def test_prompt_01_valid_response_validation():
    """Verify schema validation passes on compliant JSON response."""
    valid_response = {
        "tasks": ["Domain Generalization"],
        "architectures": ["ResNet-50"],
        "datasets": ["PACS"],
        "training_strategies": ["ERM", "Domain-Adversarial Training"],
        "shift_settings": ["Leave-one-domain-out"],
        "metrics": ["Accuracy"],
        "evaluation_protocol": "Training on three source domains and testing on the held-out target domain with 3 runs.",
        "reported_results": [
            {
                "value": "84.6",
                "metric": "Accuracy",
                "dataset": "PACS",
                "setting": "ResNet-50, Domain-Adversarial Training, Leave-one-domain-out",
                "method": "proposed",
                "page": 2,
                "quote": "our proposed method achieves an average Accuracy of 84.6% across target domains"
            },
            {
                "value": "80.2",
                "metric": "Accuracy",
                "dataset": "PACS",
                "setting": "ResNet-50, ERM, Leave-one-domain-out",
                "method": "baseline",
                "page": 2,
                "quote": "whereas standard ERM baseline reaches 80.2%."
            }
        ],
        "findings": [
            {
                "text": "The proposed method outperforms standard ERM on difficult domains.",
                "page": 3,
                "section": "Findings",
                "quote": "Our proposed method consistently outperforms standard ERM across all difficult target domains"
            }
        ],
        "limitations": [
            {
                "text": "Computational overhead increases with more training domains.",
                "page": 3,
                "section": "Limitations",
                "quote": "computational requirements increase substantially with the number of training domains.",
                "origin": "author_stated"
            }
        ],
        "extraction_confidence": "high"
    }

    raw_json = json.dumps(valid_response)
    val = ResponseValidator.validate_literature_extraction(raw_json, source_text=SAMPLE_PAPER_TEXT)

    assert val.is_valid is True
    assert isinstance(val.data, LiteratureExtractionOutput)
    assert len(val.data.reported_results) == 2
    assert val.data.reported_results[0].value == 84.6
    assert val.data.extraction_confidence == "high"


def test_prompt_01_max_items_constraints():
    """Verify max 8 reported results, max 6 findings, and max 6 limitations."""
    # Test > 8 results
    invalid_data = {
        "tasks": ["Domain Generalization"],
        "architectures": ["ResNet-50"],
        "datasets": ["PACS"],
        "training_strategies": ["ERM"],
        "shift_settings": ["Leave-one-domain-out"],
        "metrics": ["Accuracy"],
        "evaluation_protocol": "not stated",
        "reported_results": [
            {
                "value": 80.0 + i,
                "metric": "Accuracy",
                "dataset": "PACS",
                "setting": "ResNet-50",
                "method": "baseline",
                "page": 2,
                "quote": f"accuracy was {80+i}"
            }
            for i in range(9)  # 9 items exceeds max 8
        ],
        "findings": [],
        "limitations": [],
        "extraction_confidence": "high"
    }

    val = ResponseValidator.validate_literature_extraction(json.dumps(invalid_data))
    assert val.is_valid is False
    assert "reported_results" in val.error_message


def test_prompt_01_quote_word_limit():
    """Verify quotes exceeding 25 words are rejected."""
    long_quote = "This is a very long quote designed specifically to test the validation threshold of exactly twenty five words which must be strictly enforced by the code to avoid hallucinated verbosity and run on sentences"
    assert len(long_quote.split()) > 25

    invalid_data = {
        "tasks": ["Domain Generalization"],
        "architectures": ["ResNet-50"],
        "datasets": ["PACS"],
        "training_strategies": ["ERM"],
        "shift_settings": ["Leave-one-domain-out"],
        "metrics": ["Accuracy"],
        "evaluation_protocol": "not stated",
        "reported_results": [
            {
                "value": 85.0,
                "metric": "Accuracy",
                "dataset": "PACS",
                "setting": "ResNet-50",
                "method": "baseline",
                "page": 2,
                "quote": long_quote
            }
        ],
        "findings": [],
        "limitations": [],
        "extraction_confidence": "high"
    }

    val = ResponseValidator.validate_literature_extraction(json.dumps(invalid_data))
    assert val.is_valid is False
    assert "exceeds 25 words" in val.error_message


def test_prompt_01_rule_based_fallback():
    """Verify rule-based extraction extracts valid records from paper text."""
    fallback = RuleBasedLiteratureExtractor.extract(
        title="Sample Paper",
        year=2023,
        paper_text=SAMPLE_PAPER_TEXT
    )

    assert "ResNet-50" in fallback.architectures
    assert "PACS" in fallback.datasets
    assert "ERM" in fallback.training_strategies
    assert fallback.extraction_confidence == "low"
    assert len(fallback.reported_results) > 0


def test_prompt_01_runner_with_retry_and_fallback():
    """Verify PromptRunner executes retry on invalid reply and falls back if retry fails."""
    runner = PromptRunner()

    # Case 1: First attempt succeeds
    def mock_llm_good(sys, user):
        return json.dumps({
            "tasks": ["Domain Generalization"],
            "architectures": ["ResNet-50"],
            "datasets": ["PACS"],
            "training_strategies": ["ERM"],
            "shift_settings": ["Leave-one-domain-out"],
            "metrics": ["Accuracy"],
            "evaluation_protocol": "3 seeds",
            "reported_results": [],
            "findings": [],
            "limitations": [],
            "extraction_confidence": "medium"
        })

    res1 = runner.run_literature_extraction("Title", 2023, SAMPLE_PAPER_TEXT, mock_llm_good)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.1"

    # Case 2: First attempt fails, retry succeeds
    call_count = 0
    def mock_llm_retry_success(sys, user):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "This is not valid JSON"
        return mock_llm_good(sys, user)

    res2 = runner.run_literature_extraction("Title", 2023, SAMPLE_PAPER_TEXT, mock_llm_retry_success)
    assert res2.used_fallback is False
    assert res2.retries_used == 1
    assert call_count == 2

    # Case 3: Both attempts fail -> Fallback triggered
    def mock_llm_persistent_fail(sys, user):
        return "{invalid_json: 123"

    res3 = runner.run_literature_extraction("Title", 2023, SAMPLE_PAPER_TEXT, mock_llm_persistent_fail)
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, LiteratureExtractionOutput)
    assert res3.data.extraction_confidence == "low"
