"""
Unit and Validation Tests for Prompt 03: Contradiction Judgement
File: prompts/03_contradiction_judgement.md | Version 1.1 | Stage: contradictions
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import ContradictionJudgementOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedContradictionEvaluator
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_PAIR = """
Result 1:
- Paper: P01
- Reported value: 84.6
- Metric: Accuracy
- Dataset: PACS
- Setting: ResNet-50, Domain-adversarial training, Leave-one-domain-out
- Evaluation protocol: 3 random seeds, leave-one-domain-out
- Quote: "our proposed method achieves an average Accuracy of 84.6% across target domains"

Result 2:
- Paper: P02
- Reported value: 72.1
- Metric: Accuracy
- Dataset: OfficeHome
- Setting: ResNet-50, Domain-adversarial training, Leave-one-domain-out
- Evaluation protocol: 3 random seeds, leave-one-domain-out
- Quote: "adversarial domain training achieves 72.1% accuracy on OfficeHome"
"""


def test_prompt_03_metadata_and_loading():
    """Verify prompt 03 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p3 = loader.load_prompt("03_contradiction_judgement")

    assert p3.name == "03 Contradiction judgement"
    assert p3.version == "1.1"
    assert p3.stage == "contradictions"
    assert "explain whether two reported results truly conflict" in p3.purpose
    assert "careful experimental reviewer" in p3.system_prompt
    assert p3.placeholders == {"pair"}


def test_prompt_03_template_rendering():
    """Verify template placeholder replacement for {{pair}}."""
    loader = PromptLoader()
    p3 = loader.load_prompt("03_contradiction_judgement")

    rendered = p3.render_user_prompt(pair=SAMPLE_PAIR)
    assert "Result 1:" in rendered
    assert "Result 2:" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p3.render_user_prompt()


def test_prompt_03_valid_response_validation():
    """Verify schema validation on compliant contradiction judgement output."""
    valid_data = {
        "classification": "non-comparable evidence",
        "explanation": "Paper P01 evaluates performance on the PACS benchmark (84.6% accuracy), whereas Paper P02 tests on OfficeHome (72.1% accuracy). Because these benchmark datasets feature entirely distinct domain shifts and category granularities, the numeric discrepancy does not indicate an empirical contradiction.",
        "conditions_to_verify": [
            "Check domain transfer difficulties between PACS and OfficeHome",
            "Verify image resolution and data augmentation schemes",
            "Check learning rate schedules and batch sizes"
        ]
    }

    val = ResponseValidator.validate_contradiction_judgement(json.dumps(valid_data), pair_context=SAMPLE_PAIR)
    assert val.is_valid is True
    assert isinstance(val.data, ContradictionJudgementOutput)
    assert val.data.classification == "non-comparable evidence"
    assert len(val.data.conditions_to_verify) == 3


def test_prompt_03_rule_never_declare_wrong():
    """Verify rule: 'Never declare either paper wrong'."""
    invalid_data = {
        "classification": "apparent disagreement under comparable settings",
        "explanation": "Paper P01 is wrong because 84.6% is clearly overstated compared to Paper P02. The result in Paper P01 is incorrect.",
        "conditions_to_verify": ["Check baseline code"]
    }

    val = ResponseValidator.validate_contradiction_judgement(json.dumps(invalid_data))
    assert val.is_valid is False
    assert "Rule violation: Never declare either paper wrong" in val.error_message


def test_prompt_03_rule_unspecified_protocol():
    """Verify rule: 'If protocol is not stated on either side, choose insufficient info or non-comparable, never apparent disagreement'."""
    unspecified_pair = """
Result 1:
- Paper: P01
- Reported value: 85.0
- Metric: Accuracy
- Dataset: PACS
- Evaluation protocol: not stated

Result 2:
- Paper: P02
- Reported value: 78.0
- Metric: Accuracy
- Dataset: PACS
- Evaluation protocol: 3 runs
"""
    disagreement_response = {
        "classification": "apparent disagreement under comparable settings",
        "explanation": "Both studies evaluate PACS with Accuracy but achieve differing numbers of 85.0% and 78.0%.",
        "conditions_to_verify": ["Check test splits"]
    }

    val = ResponseValidator.validate_contradiction_judgement(
        json.dumps(disagreement_response),
        pair_context=unspecified_pair
    )
    assert val.is_valid is False
    assert "If the protocol, metric definition or split is not stated" in val.error_message


def test_prompt_03_max_conditions_to_verify():
    """Verify conditions_to_verify accepts at most 4 items."""
    invalid_data = {
        "classification": "apparent agreement",
        "explanation": "Both papers report matching accuracy within 1% on PACS with identical ResNet-50 configurations.",
        "conditions_to_verify": ["item 1", "item 2", "item 3", "item 4", "item 5"]  # 5 items > 4
    }

    val = ResponseValidator.validate_contradiction_judgement(json.dumps(invalid_data))
    assert val.is_valid is False
    assert "conditions_to_verify" in val.error_message


def test_prompt_03_rule_based_fallback():
    """Verify fallback evaluator correctly handles cross-dataset and unstated protocol pairs."""
    # Cross dataset pair
    fallback_ds = RuleBasedContradictionEvaluator.evaluate(SAMPLE_PAIR)
    assert fallback_ds.classification == "non-comparable evidence"
    assert "OfficeHome" in fallback_ds.explanation

    # Unstated protocol pair
    fallback_proto = RuleBasedContradictionEvaluator.evaluate("Paper 1: PACS. Protocol: not stated. Paper 2: PACS.")
    assert fallback_proto.classification == "insufficient information"


def test_prompt_03_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 03."""
    runner = PromptRunner()

    valid_json = json.dumps({
        "classification": "non-comparable evidence",
        "explanation": "Different datasets were used, meaning these results do not contradict each other.",
        "conditions_to_verify": ["Check split", "Check domain"]
    })

    # Case 1: First attempt succeeds
    res1 = runner.run_contradiction_judgement(SAMPLE_PAIR, lambda s, u: valid_json)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.1"

    # Case 2: First attempt fails, retry succeeds
    attempt = 0
    def mock_llm_retry(s, u):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "Broken format"
        return valid_json

    res2 = runner.run_contradiction_judgement(SAMPLE_PAIR, mock_llm_retry)
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_contradiction_judgement(SAMPLE_PAIR, lambda s, u: "Invalid")
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, ContradictionJudgementOutput)
