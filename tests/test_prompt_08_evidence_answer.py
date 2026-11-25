"""
Unit and Validation Tests for Prompt 08: Evidence Answer
File: prompts/08_evidence_answer.md | Version 1.1 | Stage: ask_the_papers
"""

import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import EvidenceAnswerOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedEvidenceAnswerer
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_QUESTION = "What accuracy does ERM achieve on the PACS benchmark?"
SAMPLE_PASSAGES = (
    "[1] We evaluate ResNet-50 trained with standard ERM on the PACS benchmark, reaching 80.2% average accuracy.\n"
    "[2] On OfficeHome, ERM achieves 65.4% accuracy under leave-one-domain-out evaluation."
)


def test_prompt_08_metadata_and_loading():
    """Verify prompt 08 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p8 = loader.load_prompt("08_evidence_answer")

    assert p8.name == "08 Evidence answer"
    assert p8.version == "1.1"
    assert p8.stage == "ask_the_papers"
    assert "answer a question using only the numbered passages" in p8.purpose
    assert "You reply with plain text" in p8.system_prompt
    assert p8.placeholders == {"question", "passages"}


def test_prompt_08_template_rendering():
    """Verify template placeholder replacement for {{question}} and {{passages}}."""
    loader = PromptLoader()
    p8 = loader.load_prompt("08_evidence_answer")

    rendered = p8.render_user_prompt(question=SAMPLE_QUESTION, passages=SAMPLE_PASSAGES)
    assert SAMPLE_QUESTION in rendered
    assert "[1] We evaluate ResNet-50" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p8.render_user_prompt(question=SAMPLE_QUESTION)


def test_prompt_08_valid_response_validation():
    """Verify schema validation on compliant cited answer."""
    valid_answer = "ResNet-50 trained with standard Empirical Risk Minimization (ERM) achieves an average accuracy of 80.2% on the PACS benchmark [1]."

    val = ResponseValidator.validate_evidence_answer(valid_answer, passages=SAMPLE_PASSAGES)
    assert val.is_valid is True
    assert isinstance(val.data, EvidenceAnswerOutput)
    assert val.data.is_declined is False
    assert val.data.citations == [1]


def test_prompt_08_valid_decline():
    """Verify schema validation when passages do not contain the answer."""
    decline_answer = "The passages do not answer this question."

    val = ResponseValidator.validate_evidence_answer(decline_answer, passages=SAMPLE_PASSAGES)
    assert val.is_valid is True
    assert val.data.is_declined is True


def test_prompt_08_rule_missing_citations():
    """Verify rule: Cite every claim with a passage number in square brackets."""
    uncited_answer = "ERM achieves 80.2% accuracy on the PACS benchmark across all domains."

    val = ResponseValidator.validate_evidence_answer(uncited_answer, passages=SAMPLE_PASSAGES)
    assert val.is_valid is False
    assert "Cite every claim with a passage number in square brackets" in val.error_message


def test_prompt_08_rule_unlisted_passage_number():
    """Verify rule: Only use numbers that appear in the passage list."""
    invalid_cite = "ERM achieves 80.2% accuracy on PACS [99]."  # 99 is not in passages [1] or [2]

    val = ResponseValidator.validate_evidence_answer(invalid_cite, passages=SAMPLE_PASSAGES)
    assert val.is_valid is False
    assert "Only use numbers that appear in the passage list" in val.error_message


def test_prompt_08_rule_max_four_paragraphs():
    """Verify rule: Answer in one to four short paragraphs."""
    five_paragraphs = (
        "Paragraph 1 discusses ERM on PACS [1].\n\n"
        "Paragraph 2 discusses OfficeHome [2].\n\n"
        "Paragraph 3 discusses domain shifts [1].\n\n"
        "Paragraph 4 discusses test accuracy [2].\n\n"
        "Paragraph 5 summarizes conclusions [1]."
    )

    val = ResponseValidator.validate_evidence_answer(five_paragraphs, passages=SAMPLE_PASSAGES)
    assert val.is_valid is False
    assert "one to four short paragraphs" in val.error_message


def test_prompt_08_rule_based_fallback():
    """Verify fallback answerer retrieves relevant passage or declines appropriately."""
    # Relevant question
    res_ans = RuleBasedEvidenceAnswerer.answer(SAMPLE_QUESTION, SAMPLE_PASSAGES)
    assert res_ans.is_declined is False
    assert len(res_ans.citations) > 0

    # Irrelevant question
    res_dec = RuleBasedEvidenceAnswerer.answer("What is the capital of France?", SAMPLE_PASSAGES)
    assert res_dec.is_declined is True
    assert "The passages do not answer this question" in res_dec.text


def test_prompt_08_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 08."""
    runner = PromptRunner()

    valid_answer = "ERM achieves 80.2% average accuracy on PACS [1]."

    # Case 1: First attempt succeeds
    res1 = runner.run_evidence_answer(SAMPLE_QUESTION, SAMPLE_PASSAGES, lambda s, u: valid_answer)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.1"

    # Case 2: First attempt fails (uncited), retry succeeds
    attempt = 0
    def mock_retry(s, u):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "Uncited answer without brackets."
        return valid_answer

    res2 = runner.run_evidence_answer(SAMPLE_QUESTION, SAMPLE_PASSAGES, mock_retry)
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_evidence_answer(SAMPLE_QUESTION, SAMPLE_PASSAGES, lambda s, u: "Uncited broken text")
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, EvidenceAnswerOutput)
