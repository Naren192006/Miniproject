"""
Unit and Validation Tests for Prompt 07: Report Summary
File: prompts/07_report_summary.md | Version 1.0 | Stage: report
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import ReportSummaryOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedReportSummarizer
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_FACTS = json.dumps({
    "paper_count": 3,
    "domain": "Domain Generalization",
    "benchmarks": ["PACS", "OfficeHome"],
    "candidate_gaps": ["ViT on OfficeHome under leave-one-domain-out"],
    "unrun_analyses": ["Automated code execution benchmark"],
    "declined_hypotheses": ["P03 on DomainNet without reported baseline"]
})


def test_prompt_07_metadata_and_loading():
    """Verify prompt 07 loads correctly with version 1.0 and expected metadata."""
    loader = PromptLoader()
    p7 = loader.load_prompt("07_report_summary")

    assert p7.name == "07 Report summary"
    assert p7.version == "1.0"
    assert p7.stage == "report"
    assert "short plain-language summary" in p7.purpose
    assert "reply with plain text, not JSON" in p7.system_prompt
    assert p7.placeholders == {"facts"}


def test_prompt_07_template_rendering():
    """Verify template placeholder replacement for {{facts}}."""
    loader = PromptLoader()
    p7 = loader.load_prompt("07_report_summary")

    rendered = p7.render_user_prompt(facts=SAMPLE_FACTS)
    assert "paper_count" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p7.render_user_prompt()


def test_prompt_07_valid_response_validation():
    """Verify schema validation on compliant 3-paragraph plain-text summary under 180 words."""
    valid_summary = (
        "The examined corpus evaluates domain generalization across three research papers focused on PACS and OfficeHome benchmarks. Structured extraction captured model architectures, training strategies, and empirical accuracy distributions across leave-one-domain-out shifts.\n\n"
        "Analysis revealed candidate gaps involving Vision Transformers on OfficeHome. Several candidate hypotheses were evaluated, while one candidate hypothesis was declined due to missing baseline records.\n\n"
        "Researchers should verify original evaluation protocols and check random seed variance before conducting laboratory experiments. Automated code execution could not run and requires manual validation."
    )

    val = ResponseValidator.validate_report_summary(valid_summary, facts_json=SAMPLE_FACTS)
    assert val.is_valid is True
    assert isinstance(val.data, ReportSummaryOutput)
    assert len(val.data.paragraphs) == 3
    assert val.data.word_count <= 180


def test_prompt_07_rule_no_headings():
    """Verify rule: No headings allowed in report summary."""
    invalid_heading = (
        "### Overview\n"
        "The corpus covers three papers on domain generalization.\n\n"
        "Candidate gaps include Vision Transformers.\n\n"
        "Researchers must check baseline seeds."
    )

    val = ResponseValidator.validate_report_summary(invalid_heading)
    assert val.is_valid is False
    assert "No headings allowed" in val.error_message


def test_prompt_07_rule_no_bullets():
    """Verify rule: No bullet points allowed in report summary."""
    invalid_bullets = (
        "The corpus covers three papers on domain generalization.\n\n"
        "- Candidate gap 1: ViT on OfficeHome\n"
        "- Candidate gap 2: Mixup comparison\n\n"
        "Researchers must check baseline seeds."
    )

    val = ResponseValidator.validate_report_summary(invalid_bullets)
    assert val.is_valid is False
    assert "No bullet points" in val.error_message


def test_prompt_07_rule_word_limit():
    """Verify rule: No more than 180 words."""
    long_summary = (
        ("This is a verbose paragraph describing the corpus in extensive repetitive detail to exceed the word limit. " * 8) + "\n\n" +
        ("This is the second paragraph repeating claims about candidate gaps and candidate findings in excess prose. " * 8) + "\n\n" +
        ("This is the third paragraph listing verification protocols and seed evaluations over multiple sentences. " * 8)
    )

    val = ResponseValidator.validate_report_summary(long_summary)
    assert val.is_valid is False
    assert "180" in val.error_message or "word_count" in val.error_message


def test_prompt_07_rule_must_be_three_paragraphs():
    """Verify rule: Exactly three short paragraphs."""
    two_paragraphs = (
        "The corpus covers three papers on domain generalization across benchmark datasets.\n\n"
        "Candidate gaps involve Vision Transformers and researchers should check baseline seeds."
    )

    val = ResponseValidator.validate_report_summary(two_paragraphs)
    assert val.is_valid is False
    assert "exactly three paragraphs" in val.error_message


def test_prompt_07_rule_no_discovery_claims():
    """Verify rule: Do not say that a gap is new, unexplored or a discovery."""
    invalid_discovery = (
        "The corpus covers three papers on domain generalization.\n\n"
        "We report a breakthrough discovery and an unexplored gap in vision models.\n\n"
        "Researchers should verify baselines."
    )

    val = ResponseValidator.validate_report_summary(invalid_discovery)
    assert val.is_valid is False
    assert "Do not say that a gap is new, unexplored or a discovery" in val.error_message


def test_prompt_07_rule_based_fallback():
    """Verify fallback summarizer produces compliant 3-paragraph text under 180 words."""
    summary = RuleBasedReportSummarizer.summarize(SAMPLE_FACTS)
    assert isinstance(summary, ReportSummaryOutput)
    assert len(summary.paragraphs) == 3
    assert summary.word_count <= 180
    assert "candidate" in summary.text.lower()


def test_prompt_07_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 07."""
    runner = PromptRunner()

    valid_text = (
        "The examined corpus covers three research papers examining domain generalization across PACS and OfficeHome.\n\n"
        "Candidate gaps and candidate contradictions were identified across model architectures and shift settings.\n\n"
        "Researchers should check original training protocols and random seed counts before initiating experiments."
    )

    # Case 1: First attempt succeeds
    res1 = runner.run_report_summary(SAMPLE_FACTS, lambda s, u: valid_text)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.0"

    # Case 2: First attempt fails (has bullet points), retry succeeds
    attempt = 0
    def mock_retry(s, u):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "- Bullet 1\n- Bullet 2"
        return valid_text

    res2 = runner.run_report_summary(SAMPLE_FACTS, mock_retry)
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_report_summary(SAMPLE_FACTS, lambda s, u: "Always broken")
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, ReportSummaryOutput)
