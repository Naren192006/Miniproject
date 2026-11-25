"""
Integrity Tests for the Full HypothesisForge Prompt Pack (Prompts 01 - 08)
Verifies that all 8 prompts from the prompt pack match specifications and load cleanly.
"""

import pytest
from pathlib import Path
from hypothesisforge.prompt_engine.loader import PromptLoader


EXPECTED_PROMPTS = [
    {
        "id": "01",
        "name": "01 Literature extraction",
        "file": "01_literature_extraction.md",
        "version": "1.1",
        "stage": "literature",
        "placeholders": {"title", "year", "paper_text"}
    },
    {
        "id": "02",
        "name": "02 Gap relevance",
        "file": "02_gap_relevance.md",
        "version": "1.1",
        "stage": "gaps",
        "placeholders": {"gap", "related_records"}
    },
    {
        "id": "03",
        "name": "03 Contradiction judgement",
        "file": "03_contradiction_judgement.md",
        "version": "1.1",
        "stage": "contradictions",
        "placeholders": {"pair"}
    },
    {
        "id": "04",
        "name": "04 Hypothesis generation",
        "file": "04_hypothesis_generation.md",
        "version": "1.1",
        "stage": "hypotheses",
        "placeholders": {"gap"}
    },
    {
        "id": "05",
        "name": "05 Scientific critic",
        "file": "05_scientific_critic.md",
        "version": "1.1",
        "stage": "critic",
        "placeholders": {"hypothesis", "checks", "prior_status", "overlap"}
    },
    {
        "id": "06",
        "name": "06 Experiment design",
        "file": "06_experiment_design.md",
        "version": "1.1",
        "stage": "experiments",
        "placeholders": {"hypothesis", "draft_plan", "grounded_options"}
    },
    {
        "id": "07",
        "name": "07 Report summary",
        "file": "07_report_summary.md",
        "version": "1.0",
        "stage": "report",
        "placeholders": {"facts"}
    },
    {
        "id": "08",
        "name": "08 Evidence answer",
        "file": "08_evidence_answer.md",
        "version": "1.1",
        "stage": "ask_the_papers",
        "placeholders": {"question", "passages"}
    },
]


def test_all_prompts_exist_and_load():
    loader = PromptLoader()

    for spec in EXPECTED_PROMPTS:
        p = loader.load_prompt(spec["id"])
        assert p.version == spec["version"], f"Prompt {spec['id']} version mismatch"
        assert p.stage == spec["stage"], f"Prompt {spec['id']} stage mismatch"
        assert len(p.system_prompt) > 20, f"Prompt {spec['id']} system prompt too short"
        assert len(p.user_template) > 20, f"Prompt {spec['id']} user template too short"
        assert spec["placeholders"].issubset(p.placeholders), (
            f"Prompt {spec['id']} expected placeholders {spec['placeholders']}, got {p.placeholders}"
        )
