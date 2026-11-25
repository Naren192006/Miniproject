"""
Unit and Validation Tests for Prompt 02: Gap Relevance
File: prompts/02_gap_relevance.md | Version 1.1 | Stage: gaps
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import GapRelevanceOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedGapEvaluator
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_GAP = "Vision Transformer (ViT-B/16) on OfficeHome dataset under Leave-one-domain-out shift"
SAMPLE_RECORDS = """
- Paper P01: ResNet-50 on PACS with ERM and Domain-Adversarial Training.
- Paper P02: ResNet-50 on OfficeHome with ERM and CORAL.
- Paper P03: ViT-B/16 on PACS with ERM.
"""


def test_prompt_02_metadata_and_loading():
    """Verify prompt 02 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p2 = loader.load_prompt("02_gap_relevance")

    assert p2.name == "02 Gap relevance"
    assert p2.version == "1.1"
    assert p2.stage == "gaps"
    assert "judge whether testing an untested combination would matter" in p2.purpose
    assert "You are a senior machine learning researcher" in p2.system_prompt
    assert "This is not a novelty check" in p2.purpose
    assert p2.placeholders == {"gap", "related_records"}


def test_prompt_02_template_rendering():
    """Verify placeholder substitution for candidate gap and related records."""
    loader = PromptLoader()
    p2 = loader.load_prompt("02_gap_relevance")

    rendered = p2.render_user_prompt(
        gap=SAMPLE_GAP,
        related_records=SAMPLE_RECORDS
    )

    assert SAMPLE_GAP in rendered
    assert "Paper P01: ResNet-50 on PACS" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p2.render_user_prompt(gap=SAMPLE_GAP)


def test_prompt_02_valid_response_validation():
    """Verify schema validation on compliant JSON."""
    valid_response = {
        "rating": "high",
        "reason": "Paper P02 establishes an OfficeHome baseline with ResNet-50, while Paper P03 demonstrates ViT viability on PACS. Evaluating ViT on OfficeHome addresses an untested combination directly motivated by these papers.",
        "absence_explanation": "not reported by these papers but plausible",
        "questions_to_check": [
            "Does literature outside this corpus evaluate ViT on OfficeHome?",
            "What pretraining resolution is required for ViT on OfficeHome?"
        ]
    }

    val = ResponseValidator.validate_gap_relevance(json.dumps(valid_response))
    assert val.is_valid is True
    assert isinstance(val.data, GapRelevanceOutput)
    assert val.data.rating == "high"
    assert val.data.absence_explanation == "not reported by these papers but plausible"
    assert len(val.data.questions_to_check) == 2


def test_prompt_02_rule_violation_novelty_claim():
    """Verify rule: 'Never state that the gap is novel or that nobody has tested it'."""
    invalid_response = {
        "rating": "high",
        "reason": "This gap is novel and completely unprecedented. Nobody has tested ViT on OfficeHome before in literature.",
        "absence_explanation": "not reported by these papers but plausible",
        "questions_to_check": []
    }

    val = ResponseValidator.validate_gap_relevance(json.dumps(invalid_response))
    assert val.is_valid is False
    assert "Rule violation: No prompt may claim that a finding is novel" in val.error_message


def test_prompt_02_rule_violation_thin_evidence():
    """Verify rule: 'If the evidence is too thin to judge, rate it \"low\" and say that the evidence is thin.'"""
    # Rating 'high' when reason states evidence is thin must fail validation
    invalid_response = {
        "rating": "high",
        "reason": "The evidence is thin across the corpus papers P01 and P02 to make a firm determination.",
        "absence_explanation": "corpus size",
        "questions_to_check": ["Is more data available?"]
    }

    val = ResponseValidator.validate_gap_relevance(json.dumps(invalid_response))
    assert val.is_valid is False
    assert "rating must be 'low'" in val.error_message

    # Rating 'low' when reason states evidence is thin must pass
    valid_response = {
        "rating": "low",
        "reason": "The evidence is thin across the corpus papers P01 and P02 to make a firm determination.",
        "absence_explanation": "corpus size",
        "questions_to_check": ["Is more data available?"]
    }
    val2 = ResponseValidator.validate_gap_relevance(json.dumps(valid_response))
    assert val2.is_valid is True
    assert val2.data.rating == "low"


def test_prompt_02_max_questions_limit():
    """Verify list of up to three questions to check."""
    invalid_response = {
        "rating": "medium",
        "reason": "Paper P01 and P02 report related results on PACS and OfficeHome respectively.",
        "absence_explanation": "corpus size",
        "questions_to_check": [
            "Question 1?",
            "Question 2?",
            "Question 3?",
            "Question 4?"  # 4 questions exceeds max 3
        ]
    }

    val = ResponseValidator.validate_gap_relevance(json.dumps(invalid_response))
    assert val.is_valid is False
    assert "questions_to_check" in val.error_message


def test_prompt_02_rule_based_fallback():
    """Verify fallback evaluator generates compliant outputs for thin and standard corpora."""
    # Thin evidence
    res_thin = RuleBasedGapEvaluator.evaluate(SAMPLE_GAP, "Paper P01: ResNet on PACS")
    assert res_thin.rating == "low"
    assert "thin" in res_thin.reason.lower()
    assert res_thin.absence_explanation == "corpus size"
    assert len(res_thin.questions_to_check) <= 3

    # Normal evidence
    res_normal = RuleBasedGapEvaluator.evaluate(SAMPLE_GAP, SAMPLE_RECORDS)
    assert res_normal.rating == "medium"
    assert res_normal.absence_explanation == "not reported by these papers but plausible"


def test_prompt_02_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 02."""
    runner = PromptRunner()

    # Case 1: First attempt succeeds
    def mock_llm_good(sys, user):
        return json.dumps({
            "rating": "medium",
            "reason": "Paper P01 and P02 evaluate ResNet-50, suggesting ViT-B/16 is a plausible untested candidate.",
            "absence_explanation": "not reported by these papers but plausible",
            "questions_to_check": ["Check ImageNet pretraining weights"]
        })

    res1 = runner.run_gap_relevance(SAMPLE_GAP, SAMPLE_RECORDS, mock_llm_good)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.1"

    # Case 2: Novelty violation on 1st call, corrected on retry
    call_count = 0
    def mock_llm_novelty_retry(sys, user):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({
                "rating": "high",
                "reason": "This gap is novel and nobody has tested it before.",
                "absence_explanation": "unknown",
                "questions_to_check": []
            })
        return mock_llm_good(sys, user)

    res2 = runner.run_gap_relevance(SAMPLE_GAP, SAMPLE_RECORDS, mock_llm_novelty_retry)
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Persistent failure triggers fallback
    def mock_llm_bad(sys, user):
        return "Not JSON"

    res3 = runner.run_gap_relevance(SAMPLE_GAP, SAMPLE_RECORDS, mock_llm_bad)
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, GapRelevanceOutput)
