"""
Unit and Validation Tests for Prompt 04: Hypothesis Generation
File: prompts/04_hypothesis_generation.md | Version 1.1 | Stage: hypotheses
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import HypothesisGenerationOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedHypothesisGenerator
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_GAP_WITH_EVIDENCE = """
Gap: ViT-B/16 on OfficeHome dataset under Leave-one-domain-out shift
Evidence:
- Paper P01: ResNet-50 on PACS achieves 84.6% Accuracy with ERM and Domain-Adversarial Training.
- Paper P02: ResNet-50 on OfficeHome achieves 72.1% Accuracy with ERM.
- Paper P03: ViT-B/16 on PACS achieves 86.2% Accuracy, showing self-attention benefits on out-of-distribution shifts.
"""


def test_prompt_04_metadata_and_loading():
    """Verify prompt 04 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p4 = loader.load_prompt("04_hypothesis_generation")

    assert p4.name == "04 Hypothesis generation"
    assert p4.version == "1.1"
    assert p4.stage == "hypotheses"
    assert "write one precise, testable hypothesis" in p4.purpose
    assert "falsifiable hypotheses" in p4.system_prompt
    assert p4.placeholders == {"gap"}


def test_prompt_04_template_rendering():
    """Verify template placeholder replacement for {{gap}}."""
    loader = PromptLoader()
    p4 = loader.load_prompt("04_hypothesis_generation")

    rendered = p4.render_user_prompt(gap=SAMPLE_GAP_WITH_EVIDENCE)
    assert "ViT-B/16 on OfficeHome" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p4.render_user_prompt()


def test_prompt_04_valid_hypothesis_validation():
    """Verify schema validation on a compliant testable hypothesis."""
    valid_data = {
        "decision": "hypothesis",
        "statement": "If ViT-B/16 is trained on OfficeHome with ERM, then out-of-domain Accuracy will increase, measured by Accuracy, relative to ResNet-50 ERM.",
        "expected_direction": "increase",
        "baseline": "ResNet-50 ERM on OfficeHome",
        "metric": "Accuracy",
        "assumptions": [
            "ViT-B/16 uses standard ImageNet-21k pretraining",
            "Leave-one-domain-out evaluation protocol is maintained",
            "Data augmentation matches standard domain generalization benchmarks"
        ],
        "cited_paper_ids": ["P02", "P03"],
        "reason": "Paper P03 demonstrates ViT superiority on PACS, while Paper P02 provides the baseline on OfficeHome."
    }

    val = ResponseValidator.validate_hypothesis_generation(
        json.dumps(valid_data),
        evidence_context=SAMPLE_GAP_WITH_EVIDENCE
    )
    assert val.is_valid is True
    assert isinstance(val.data, HypothesisGenerationOutput)
    assert val.data.decision == "hypothesis"
    assert val.data.expected_direction == "increase"
    assert len(val.data.cited_paper_ids) == 2


def test_prompt_04_valid_decline_validation():
    """Verify schema validation when declining due to insufficient evidence."""
    valid_decline = {
        "decision": "insufficient_evidence",
        "reason": "No paper in the provided corpus reports a baseline method or evaluation on the OfficeHome benchmark."
    }

    val = ResponseValidator.validate_hypothesis_generation(json.dumps(valid_decline))
    assert val.is_valid is True
    assert val.data.decision == "insufficient_evidence"


def test_prompt_04_rule_statement_format():
    """Verify statement must follow 'If ..., then ..., measured by ..., relative to ...'."""
    invalid_statement = {
        "decision": "hypothesis",
        "statement": "ViT-B/16 will achieve higher performance than ResNet-50 on OfficeHome.",
        "expected_direction": "increase",
        "baseline": "ResNet-50",
        "metric": "Accuracy",
        "assumptions": [],
        "cited_paper_ids": ["P01"],
        "reason": "Paper P01 shows good results."
    }

    val = ResponseValidator.validate_hypothesis_generation(json.dumps(invalid_statement))
    assert val.is_valid is False
    assert "Statement must be in the form 'If ..., then ..., measured by ..., relative to ...'" in val.error_message


def test_prompt_04_rule_no_novelty_claim():
    """Verify rule: 'Do not claim the hypothesis is novel'."""
    invalid_novelty = {
        "decision": "hypothesis",
        "statement": "If ViT-B/16 is applied to OfficeHome, then performance increases, measured by Accuracy, relative to ResNet-50.",
        "expected_direction": "increase",
        "baseline": "ResNet-50",
        "metric": "Accuracy",
        "assumptions": [],
        "cited_paper_ids": ["P01"],
        "reason": "This is a completely novel hypothesis that has never been tested in machine learning."
    }

    val = ResponseValidator.validate_hypothesis_generation(json.dumps(invalid_novelty))
    assert val.is_valid is False
    assert "Rule violation: Do not claim the hypothesis is novel" in val.error_message


def test_prompt_04_rule_vague_improve_verb():
    """Verify rule: 'Avoid vague verbs such as improve unless the metric is named'."""
    invalid_improve = {
        "decision": "hypothesis",
        "statement": "If ViT-B/16 is applied, then results will improve, measured by Accuracy, relative to ResNet-50.",
        "expected_direction": "increase",
        "baseline": "ResNet-50",
        "metric": "Top-1 Accuracy",  # "Top-1 Accuracy" is not named in the statement ("Accuracy" alone was written)
        "assumptions": [],
        "cited_paper_ids": ["P01"],
        "reason": "Paper P01 supports this."
    }

    val = ResponseValidator.validate_hypothesis_generation(json.dumps(invalid_improve))
    assert val.is_valid is False
    assert "unless the metric ('Top-1 Accuracy') is named in the statement" in val.error_message


def test_prompt_04_rule_cite_only_evidence_papers():
    """Verify rule: 'you cite only paper IDs that appear in that evidence'."""
    invalid_cite = {
        "decision": "hypothesis",
        "statement": "If ViT-B/16 is evaluated, then Accuracy will increase, measured by Accuracy, relative to ResNet-50.",
        "expected_direction": "increase",
        "baseline": "ResNet-50",
        "metric": "Accuracy",
        "assumptions": [],
        "cited_paper_ids": ["P99_NON_EXISTENT"],
        "reason": "Paper P99 establishes this theoretical bound."
    }

    val = ResponseValidator.validate_hypothesis_generation(
        json.dumps(invalid_cite),
        evidence_context=SAMPLE_GAP_WITH_EVIDENCE
    )
    assert val.is_valid is False
    assert "Citing paper IDs that do not appear in the evidence: ['P99_NON_EXISTENT']" in val.error_message


def test_prompt_04_rule_based_fallback():
    """Verify fallback hypothesis generator handles evidence parsing and declining."""
    # When evidence has paper IDs
    res = RuleBasedHypothesisGenerator.generate(SAMPLE_GAP_WITH_EVIDENCE)
    assert res.decision == "hypothesis"
    assert "P01" in res.cited_paper_ids or "P02" in res.cited_paper_ids
    assert "If " in res.statement and "measured by " in res.statement

    # When no paper IDs exist
    res_empty = RuleBasedHypothesisGenerator.generate("Untested gap with no papers")
    assert res_empty.decision == "insufficient_evidence"


def test_prompt_04_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 04."""
    runner = PromptRunner()

    valid_json = json.dumps({
        "decision": "hypothesis",
        "statement": "If ViT-B/16 is evaluated on OfficeHome, then Accuracy will increase, measured by Accuracy, relative to ResNet-50.",
        "expected_direction": "increase",
        "baseline": "ResNet-50 ERM",
        "metric": "Accuracy",
        "assumptions": ["Standard weights"],
        "cited_paper_ids": ["P01", "P02"],
        "reason": "Supported by Paper P01 and P02 evidence."
    })

    # Case 1: First attempt succeeds
    res1 = runner.run_hypothesis_generation(SAMPLE_GAP_WITH_EVIDENCE, lambda s, u: valid_json)
    assert res1.used_fallback is False
    assert res1.retries_used == 0
    assert res1.prompt_version == "1.1"

    # Case 2: First attempt fails, retry succeeds
    attempt = 0
    def mock_retry(s, u):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return "Broken JSON"
        return valid_json

    res2 = runner.run_hypothesis_generation(SAMPLE_GAP_WITH_EVIDENCE, mock_retry)
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_hypothesis_generation(SAMPLE_GAP_WITH_EVIDENCE, lambda s, u: "Always broken")
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, HypothesisGenerationOutput)
