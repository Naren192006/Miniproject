"""
Unit and Validation Tests for Prompt 06: Experiment Design
File: prompts/06_experiment_design.md | Version 1.1 | Stage: experiments
"""

import json
import pytest

from hypothesisforge.prompt_engine.loader import PromptLoader
from hypothesisforge.prompt_engine.validator import ResponseValidator
from hypothesisforge.prompt_engine.models import ExperimentDesignOutput
from hypothesisforge.prompt_engine.fallbacks import RuleBasedExperimentDesigner
from hypothesisforge.prompt_engine.runner import PromptRunner


SAMPLE_HYPOTHESIS = (
    "If ViT-B/16 is evaluated on OfficeHome, then out-of-domain Accuracy will increase, "
    "measured by Accuracy, relative to ResNet-50 ERM."
)
SAMPLE_DRAFT_PLAN = "Evaluate ViT-B/16 against ResNet-50 ERM on OfficeHome."
SAMPLE_GROUNDED_OPTIONS = "Datasets: OfficeHome, PACS. Baselines: ResNet-50 ERM."


def test_prompt_06_metadata_and_loading():
    """Verify prompt 06 loads correctly with version 1.1 and expected metadata."""
    loader = PromptLoader()
    p6 = loader.load_prompt("06_experiment_design")

    assert p6.name == "06 Experiment design"
    assert p6.version == "1.1"
    assert p6.stage == "experiments"
    assert "refine a draft experiment plan" in p6.purpose
    assert "experimental methodologist" in p6.system_prompt
    assert p6.placeholders == {"hypothesis", "draft_plan", "grounded_options"}


def test_prompt_06_template_rendering():
    """Verify placeholder replacement for {{hypothesis}}, {{draft_plan}}, and {{grounded_options}}."""
    loader = PromptLoader()
    p6 = loader.load_prompt("06_experiment_design")

    rendered = p6.render_user_prompt(
        hypothesis=SAMPLE_HYPOTHESIS,
        draft_plan=SAMPLE_DRAFT_PLAN,
        grounded_options=SAMPLE_GROUNDED_OPTIONS
    )

    assert "ViT-B/16 is evaluated on OfficeHome" in rendered
    assert "Evaluate ViT-B/16 against ResNet-50" in rendered
    assert "Datasets: OfficeHome, PACS" in rendered
    assert "{{" not in rendered and "}}" not in rendered

    with pytest.raises(ValueError, match="Missing required placeholder"):
        p6.render_user_prompt(hypothesis=SAMPLE_HYPOTHESIS)


def test_prompt_06_valid_response_validation():
    """Verify schema validation on a compliant experiment design output."""
    valid_data = {
        "research_question": "Does ViT-B/16 improve out-of-domain Accuracy on OfficeHome relative to ResNet-50 ERM?",
        "independent_variable": "Model architecture backbone (ViT-B/16 vs ResNet-50)",
        "dependent_variable": "Out-of-domain Accuracy (%)",
        "dataset_and_split": "OfficeHome using leave-one-domain-out protocol across 4 domains.",
        "baselines": ["ResNet-50 ERM"],
        "controls": ["AdamW optimizer", "Identical learning rate schedule", "Equal compute budget"],
        "metrics": ["Average Accuracy", "Worst-domain Accuracy"],
        "procedure": [
            "1. Download OfficeHome benchmark splits.",
            "2. Train ResNet-50 ERM across each held-out domain for 3 seeds.",
            "3. Train ViT-B/16 under identical training budgets for 3 seeds.",
            "4. Compute average and worst-domain test accuracies.",
            "5. Perform paired statistical significance tests."
        ],
        "seeds_and_runs": "Three random seeds (seeds 10, 20, 30) repeated across all 4 held-out domains.",
        "statistics": "Two-tailed paired t-test with alpha=0.05 assuming normally distributed seed performance.",
        "compute_estimate": "Estimated 40 GPU hours on an NVIDIA RTX 4090 (estimate).",
        "threats_to_validity": [
            "Confounder: Difference in parameter scale between backbones.",
            "Tuning budget: Unequal hyperparameter sweeps across methods."
        ],
        "expected_observations": "Higher accuracy on target domains indicates attention mechanisms capture more robust cross-domain visual primitives, whereas parity suggests standard inductive bias suffices.",
        "limitations": ["Constrained to OfficeHome benchmark shifts."]
    }

    val = ResponseValidator.validate_experiment_design(
        json.dumps(valid_data),
        grounded_options=SAMPLE_GROUNDED_OPTIONS
    )
    assert val.is_valid is True
    assert isinstance(val.data, ExperimentDesignOutput)
    assert len(val.data.procedure) == 5
    assert "Three random seeds" in val.data.seeds_and_runs


def test_prompt_06_rule_ungrounded_dataset_prefix():
    """Verify rule: If dataset is not in grounded list, begin with 'SUGGESTION, VERIFY:'."""
    invalid_data = {
        "research_question": "Does ViT-B/16 generalize to CIFAR-10-C?",
        "independent_variable": "Architecture",
        "dependent_variable": "Accuracy",
        "dataset_and_split": "CIFAR-10-C with corruption severity level 5.",  # Not grounded and lacks prefix
        "baselines": ["ResNet-50 ERM"],
        "controls": [],
        "metrics": ["Accuracy"],
        "procedure": ["Step 1", "Step 2"],
        "seeds_and_runs": "3 random seeds evaluated.",
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "10 hours (estimate).",
        "threats_to_validity": [],
        "expected_observations": "Higher accuracy indicates robustness.",
        "limitations": []
    }

    val = ResponseValidator.validate_experiment_design(
        json.dumps(invalid_data),
        grounded_options=SAMPLE_GROUNDED_OPTIONS
    )
    assert val.is_valid is False
    assert "begin with 'SUGGESTION, VERIFY:'" in val.error_message


def test_prompt_06_rule_ungrounded_baseline_marking():
    """Verify rule: Mark any baseline not in the grounded list with '(verify)'."""
    invalid_data = {
        "research_question": "Does ViT-B/16 improve accuracy on OfficeHome?",
        "independent_variable": "Architecture",
        "dependent_variable": "Accuracy",
        "dataset_and_split": "OfficeHome standard split.",
        "baselines": ["ResNet-50 ERM", "DANN"],  # DANN not in grounded options and not marked with (verify)
        "controls": [],
        "metrics": ["Accuracy"],
        "procedure": ["Step 1", "Step 2"],
        "seeds_and_runs": "3 random seeds evaluated.",
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "10 hours (estimate).",
        "threats_to_validity": [],
        "expected_observations": "Higher accuracy indicates robustness.",
        "limitations": []
    }

    val = ResponseValidator.validate_experiment_design(
        json.dumps(invalid_data),
        grounded_options=SAMPLE_GROUNDED_OPTIONS
    )
    assert val.is_valid is False
    assert "Mark any baseline not in the grounded list with '(verify)'" in val.error_message


def test_prompt_06_rule_procedure_max_seven_steps():
    """Verify rule: procedure list of ordered step strings, at most seven."""
    invalid_data = {
        "research_question": "Does ViT-B/16 improve accuracy on OfficeHome?",
        "independent_variable": "Architecture",
        "dependent_variable": "Accuracy",
        "dataset_and_split": "OfficeHome standard split.",
        "baselines": ["ResNet-50 ERM"],
        "controls": [],
        "metrics": ["Accuracy"],
        "procedure": [f"Step {i}" for i in range(1, 9)],  # 8 steps > 7
        "seeds_and_runs": "3 random seeds evaluated.",
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "10 hours (estimate).",
        "threats_to_validity": [],
        "expected_observations": "Higher accuracy indicates robustness.",
        "limitations": []
    }

    val = ResponseValidator.validate_experiment_design(json.dumps(invalid_data))
    assert val.is_valid is False
    assert "procedure" in val.error_message


def test_prompt_06_rule_minimum_three_seeds():
    """Verify rule: seeds_and_runs stating the number of random seeds (at least three)."""
    invalid_seeds = {
        "research_question": "Does ViT-B/16 improve accuracy on OfficeHome?",
        "independent_variable": "Architecture",
        "dependent_variable": "Accuracy",
        "dataset_and_split": "OfficeHome standard split.",
        "baselines": ["ResNet-50 ERM"],
        "controls": [],
        "metrics": ["Accuracy"],
        "procedure": ["Step 1", "Step 2"],
        "seeds_and_runs": "Only 2 random seeds used for evaluation.",  # 2 seeds < 3
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "10 hours (estimate).",
        "threats_to_validity": [],
        "expected_observations": "Higher accuracy indicates robustness.",
        "limitations": []
    }

    val = ResponseValidator.validate_experiment_design(json.dumps(invalid_seeds))
    assert val.is_valid is False
    assert "at least three random seeds" in val.error_message


def test_prompt_06_rule_no_facts_in_expectations():
    """Verify rule: Do not state an expected result as fact."""
    invalid_fact = {
        "research_question": "Does ViT-B/16 improve accuracy on OfficeHome?",
        "independent_variable": "Architecture",
        "dependent_variable": "Accuracy",
        "dataset_and_split": "OfficeHome standard split.",
        "baselines": ["ResNet-50 ERM"],
        "controls": [],
        "metrics": ["Accuracy"],
        "procedure": ["Step 1", "Step 2"],
        "seeds_and_runs": "3 random seeds evaluated.",
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "10 hours (estimate).",
        "threats_to_validity": [],
        "expected_observations": "This experiment will definitely prove that ViT is superior.",  # Fact assertion
        "limitations": []
    }

    val = ResponseValidator.validate_experiment_design(json.dumps(invalid_fact))
    assert val.is_valid is False
    assert "Do not state an expected result as fact" in val.error_message


def test_prompt_06_rule_based_fallback():
    """Verify fallback experiment designer creates a valid, compliant plan."""
    plan = RuleBasedExperimentDesigner.design(
        hypothesis=SAMPLE_HYPOTHESIS,
        draft_plan=SAMPLE_DRAFT_PLAN,
        grounded_options=SAMPLE_GROUNDED_OPTIONS
    )

    assert isinstance(plan, ExperimentDesignOutput)
    assert len(plan.procedure) <= 7
    assert "random seeds" in plan.seeds_and_runs.lower()
    assert "unknown" in plan.compute_estimate.lower()


def test_prompt_06_runner_with_retry_and_fallback():
    """Verify PromptRunner manages retry and fallback for Prompt 06."""
    runner = PromptRunner()

    valid_json = json.dumps({
        "research_question": "Does ViT-B/16 improve out-of-domain Accuracy on OfficeHome relative to ResNet-50 ERM?",
        "independent_variable": "Architecture backbone",
        "dependent_variable": "Accuracy (%)",
        "dataset_and_split": "OfficeHome standard split.",
        "baselines": ["ResNet-50 ERM"],
        "controls": ["Fixed seed schedule"],
        "metrics": ["Accuracy"],
        "procedure": ["Step 1: Download data", "Step 2: Train baseline", "Step 3: Train ViT", "Step 4: Compare"],
        "seeds_and_runs": "3 random seeds evaluated.",
        "statistics": "Paired t-test with alpha=0.05.",
        "compute_estimate": "20 GPU hours (estimate).",
        "threats_to_validity": ["Model size disparity"],
        "expected_observations": "Higher accuracy suggests attention captures generalizable features.",
        "limitations": ["Limited to OfficeHome benchmark"]
    })

    # Case 1: First attempt succeeds
    res1 = runner.run_experiment_design(
        hypothesis=SAMPLE_HYPOTHESIS,
        draft_plan=SAMPLE_DRAFT_PLAN,
        grounded_options=SAMPLE_GROUNDED_OPTIONS,
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

    res2 = runner.run_experiment_design(
        hypothesis=SAMPLE_HYPOTHESIS,
        draft_plan=SAMPLE_DRAFT_PLAN,
        grounded_options=SAMPLE_GROUNDED_OPTIONS,
        llm_fn=mock_retry
    )
    assert res2.used_fallback is False
    assert res2.retries_used == 1

    # Case 3: Both fail -> fallback
    res3 = runner.run_experiment_design(
        hypothesis=SAMPLE_HYPOTHESIS,
        draft_plan=SAMPLE_DRAFT_PLAN,
        grounded_options=SAMPLE_GROUNDED_OPTIONS,
        llm_fn=lambda s, u: "Always broken"
    )
    assert res3.used_fallback is True
    assert res3.retries_used == 1
    assert isinstance(res3.data, ExperimentDesignOutput)
