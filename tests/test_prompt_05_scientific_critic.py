"""
Unit and Validation Tests for Prompt 05: Scientific Critic
File: prompts/05_scientific_critic.md | Version 1.1 | Stage: critic
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import ScientificCriticOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedScientificCritic
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_HYPOTHESIS = (
    "If ViT-B/16 is trained on OfficeHome with ERM, then out-of-domain Accuracy will increase, "
    "measured by Accuracy, relative to ResNet-50 ERM."
)
SAMPLE_CHECKS = "Traceability: passed. Falsifiability: passed. Baseline: ResNet-50 ERM identified in P02."
SAMPLE_PRIOR_STATUS = "completed"
SAMPLE_OVERLAP = "None found"


def test_prompt_05_metadata_and_loading():
    """Verify prompt 05 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p5 = loader.load_prompt("05_scientific_critic")

    assert p5.name == "05 Scientific critic"
    assert p5.version == "1.1"
    assert p5.stage == "critic"
    assert "find weaknesses in a hypothesis" in p5.purpose
    assert "strict scientific critic" in p5.system_prompt
    assert p5.placeholders == {"hypothesis", "checks", "prior_status", "overlap"}


def test_prompt_05_template_rendering():
    """Verify placeholder replacement for all 4 variables."""
    loader = PromptLoader()
    p5 = loader.load_prompt("05_scientific_critic")

    rendered = p5.render_user_prompt(
        hypothesis=SAMPLE_HYPOTHESIS,
        checks=SAMPLE_CHECKS,
        prior_status=SAMPLE_PRIOR_STATUS,
        overlap=SAMPLE_OVERLAP
    )

    assert "ViT-B/16 is trained on OfficeHome" in rendered
    assert "Traceability: passed" in rendered
    assert "completed" in rendered
    assert "None found" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p5.render_user_prompt(hypothesis=SAMPLE_HYPOTHESIS)


def test_prompt_05_valid_accept_validation():
    """Verify schema validation on a compliant accept verdict."""
    valid_data = {
        "verdict": "accept",
        "rationale": "The hypothesis is specific, measurable, and falsifiable with a clearly grounded baseline. The prior-work search completed without finding direct overlapping studies, and traceability checks passed.",
        "missing_evidence": [],
        "alternative_explanations": [
            "Confounder: Difference in pretraining corpus scale between ViT and ResNet-50",
            "Confounder: Patch embedding architecture creates inductive bias well-suited to OfficeHome Clipart"
        ]
    }

    val = ResponseValidator.validate_scientific_critic(
        json.dumps(valid_data),
        prior_status=SAMPLE_PRIOR_STATUS,
        overlap=SAMPLE_OVERLAP,
        checks=SAMPLE_CHECKS
    )
    assert val.is_valid is True
    assert isinstance(val.data, ScientificCriticOutput)
    assert val.data.verdict == "accept"
    assert len(val.data.alternative_explanations) == 2


def test_prompt_05_rule_prior_search_not_run():
    """Verify rule: Choose 'revise' when prior-work search was not run (cannot accept)."""
    invalid_accept = {
        "verdict": "accept",
        "rationale": "The hypothesis is well written and plausible based on the evidence provided in P02 and P03.",
        "missing_evidence": [],
        "alternative_explanations": []
    }

    val = ResponseValidator.validate_scientific_critic(
        json.dumps(invalid_accept),
        prior_status="not run",
        overlap="Not searched",
        checks=SAMPLE_CHECKS
    )
    assert val.is_valid is False
    assert "Choose 'revise' when the prior-work search was not run" in val.error_message


def test_prompt_05_rule_prior_work_overlap():
    """Verify rule: Choose 'accept' only when no listed prior work covers it."""
    invalid_accept = {
        "verdict": "accept",
        "rationale": "The hypothesis is clearly formulated and supported by the corpus evidence.",
        "missing_evidence": [],
        "alternative_explanations": []
    }

    val = ResponseValidator.validate_scientific_critic(
        json.dumps(invalid_accept),
        prior_status="completed",
        overlap="Found: 'Evaluating Vision Transformers on OfficeHome Benchmark (2023)'",
        checks=SAMPLE_CHECKS
    )
    assert val.is_valid is False
    assert "Choose 'accept' only when no listed prior work covers it" in val.error_message


def test_prompt_05_rule_untraced_claim():
    """Verify rule: Choose 'reject' when the claim cannot be traced to the evidence."""
    invalid_accept = {
        "verdict": "accept",
        "rationale": "The hypothesis sounds scientifically interesting and novel.",
        "missing_evidence": [],
        "alternative_explanations": []
    }

    val = ResponseValidator.validate_scientific_critic(
        json.dumps(invalid_accept),
        prior_status="completed",
        overlap="None",
        checks="Traceability: failed, claim cannot be traced to evidence"
    )
    assert val.is_valid is False
    assert "Choose 'reject' when the claim cannot be traced to the evidence" in val.error_message


def test_prompt_05_rule_no_novelty_claim():
    """Verify rule: Claim is not verified as globally novel."""
    invalid_novelty = {
        "verdict": "accept",
        "rationale": "This hypothesis is accepted because it is globally novel and unprecedented in the entire research field.",
        "missing_evidence": [],
        "alternative_explanations": []
    }

    val = ResponseValidator.validate_scientific_critic(json.dumps(invalid_novelty))
    assert val.is_valid is False
    assert "claim is not verified as globally novel" in val.error_message


def test_prompt_05_max_lists_limit():
    """Verify maximum of 4 missing evidence items and 4 alternative explanations."""
    invalid_items = {
        "verdict": "revise",
        "rationale": "More evidence is required across multiple benchmarks before accepting.",
        "missing_evidence": ["E1", "E2", "E3", "E4", "E5"],  # 5 items > 4
        "alternative_explanations": []
    }

    val = ResponseValidator.validate_scientific_critic(json.dumps(invalid_items))
    assert val.is_valid is False
    assert "missing_evidence" in val.error_message


def test_prompt_05_rule_based_fallback():
    """Verify rule-based critic correctly returns accept, revise, or reject based on conditions."""
    # Clean -> accept
    res_clean = RuleBasedScientificCritic.evaluate(SAMPLE_HYPOTHESIS, SAMPLE_CHECKS, "completed", "None")
    assert res_clean.verdict == "accept"

    # Prior not run -> revise
    res_revise = RuleBasedScientificCritic.evaluate(SAMPLE_HYPOTHESIS, SAMPLE_CHECKS, "not run", "None")
    assert res_revise.verdict == "revise"

    # Untraced -> reject
    res_reject = RuleBasedScientificCritic.evaluate(SAMPLE_HYPOTHESIS, "Traceability: failed, cannot be traced", "completed", "None")
    assert res_reject.verdict == "reject"


def test_prompt_05_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 05."""
    runner = PromptRunner()

    valid_json = json.dumps({
        "verdict": "revise",
        "rationale": "The hypothesis is plausible but requires verification against concurrent preprints. The evaluation protocol details need confirmation.",
        "missing_evidence": ["Full preprint audit"],
        "alternative_explanations": ["Confounder: Pretraining dataset scale"]
    })

    # Case 1: First attempt succeeds
    res1 = runner.run_scientific_critic(
        hypothesis=SAMPLE_HYPOTHESIS,
        checks=SAMPLE_CHECKS,
        prior_status=SAMPLE_PRIOR_STATUS,
        overlap=SAMPLE_OVERLAP,
        llm_fn=lambda s, u: valid_json
    )
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

    res2 = runner.run_scientific_critic(
        hypothesis=SAMPLE_HYPOTHESIS,
        checks=SAMPLE_CHECKS,
        prior_status=SAMPLE_PRIOR_STATUS,
        overlap=SAMPLE_OVERLAP,
        llm_fn=mock_retry
    )
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_scientific_critic(
        hypothesis=SAMPLE_HYPOTHESIS,
        checks=SAMPLE_CHECKS,
        prior_status=SAMPLE_PRIOR_STATUS,
        overlap=SAMPLE_OVERLAP,
        llm_fn=lambda s, u: "Always broken"
    )
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, ScientificCriticOutput)
