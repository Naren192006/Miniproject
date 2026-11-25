"""
Regression & Verification Tests for Critical Review Fixes:
1. Stable Paper IDs & Topic Relevance Filtering
2. Metric-aware value extraction tested on 5 hand-checked papers
3. Missing-data handling in Gap Matrix (no Unspecified gaps, no cross-product strategy copying)
4. Honest audit fields (prompt_used=False when no LLM) & stale corpus signature
5. Hypothesis direction='unknown', non-CNN baseline, specific citations, and Critic search gate
6. Experiment plan grounded splits, unknown compute estimate, and low-power statistical note
"""

from hypothesisforge.ingestion import make_stable_paper_id
from hypothesisforge.paper_search import score_topic_relevance
from hypothesisforge.pipeline import (
    answer_workspace_question,
    build_experimental_matrix,
    compute_corpus_signature,
    get_sample_corpus,
    run_full_pipeline,
)
from hypothesisforge.prompt_engine.fallbacks import (
    RuleBasedExperimentDesigner,
    RuleBasedHypothesisGenerator,
    RuleBasedLiteratureExtractor,
    RuleBasedScientificCritic,
)
from hypothesisforge.prompt_engine.models import LiteratureExtractionOutput


def test_stable_paper_ids_and_relevance_filter():
    id1 = make_stable_paper_id("ERM++: An Improved Baseline for Domain Generalization")
    id2 = make_stable_paper_id("ERM++: An Improved Baseline for Domain Generalization")
    assert id1 == id2
    assert id1.startswith("P-")

    # Off-topic time series paper gets filtered out for Domain Generalization
    off_topic = score_topic_relevance(
        "Domain Generalization Vision Transformers",
        "Time Series Prediction under Distribution Shift using Differentiable Forgetting",
        "Financial time series forecasting under concept drift.",
    )
    on_topic = score_topic_relevance(
        "Domain Generalization Vision Transformers",
        "Vision Transformers for Domain Generalization",
        "We evaluate ViT-B/16 on PACS and OfficeHome.",
    )
    assert off_topic < 0.45
    assert on_topic >= 0.45


def test_metric_aware_extraction_on_five_hand_checked_papers():
    """
    Hand-checked 5-paper suite verifying significance thresholds, train/val splits,
    and error bars are NOT extracted as Accuracy results, while real metric sentences ARE.
    """
    hand_checked_papers = [
        # Paper 1: Contains "5% level" significance trap -> should extract 0 results
        (
            "Time Series Paper",
            "[Page 1]: A star is placed by a model's MSE whenever its MSE is significantly larger, at the 5% level, than the best model.",
            0,
            [],
        ),
        # Paper 2: Contains "80%/20% train/val split" trap -> should extract 0 results
        (
            "Blood Cell Paper",
            "[Page 1]: We split the data into training (80% of the images) and validation (20% of the images). ResNet-50 and CNN are compared with ERM.",
            0,
            [],
        ),
        # Paper 3: Contains real PACS result -> should extract 84.6% on PACS
        (
            "DANN Paper",
            "[Page 2]: On PACS, ResNet-50 with Domain-adversarial training achieves 84.6% Accuracy.",
            1,
            [84.6],
        ),
        # Paper 4: Contains real OfficeHome + Mixup results -> should extract 66.8% and 67.9%
        (
            "OfficeHome Paper",
            "[Page 2]: Standard ERM achieves 66.8% Accuracy on OfficeHome. Mixup reaches 67.9% Accuracy on OfficeHome.",
            2,
            [66.8, 67.9],
        ),
        # Paper 5: Contains real ViT result + error bar noise -> should extract 86.2%, ignore ± 0.4%
        (
            "ViT Paper",
            "[Page 2]: ViT-B/16 achieves 86.2% Accuracy on PACS. The variance margin was ± 0.4%.",
            1,
            [86.2],
        ),
    ]

    for title, text, expected_count, expected_vals in hand_checked_papers:
        out = RuleBasedLiteratureExtractor.extract(title, 2024, text)
        assert len(out.reported_results) == expected_count, f"Failed count on {title}: {out.reported_results}"
        extracted_vals = [r.value for r in out.reported_results]
        assert extracted_vals == expected_vals, f"Failed values on {title}: {extracted_vals}"


def test_missing_data_excluded_from_gap_matrix_and_no_strategy_copying():
    records = {
        "P-000001": LiteratureExtractionOutput(
            tasks=["Domain Generalization"],
            architectures=["CNN"],
            datasets=[],  # Missing dataset!
            training_strategies=["ERM"],
            shift_settings=[],
            metrics=[],
            evaluation_protocol="not stated",
            reported_results=[],
            findings=[],
            limitations=[],
            extraction_confidence="low",
        ),
        "P-000002": LiteratureExtractionOutput(
            tasks=["Domain Generalization"],
            architectures=[],  # Missing architecture!
            datasets=["Camelyon17"],
            training_strategies=[],
            shift_settings=[],
            metrics=[],
            evaluation_protocol="not stated",
            reported_results=[],
            findings=[],
            limitations=[],
            extraction_confidence="low",
        ),
        "P-000003": LiteratureExtractionOutput(
            tasks=["Domain Generalization"],
            architectures=["ResNet-50"],
            datasets=["PACS"],
            training_strategies=["ERM", "Mixup", "SWAD"],
            shift_settings=[],
            metrics=["Accuracy"],
            evaluation_protocol="not stated",
            reported_results=[],
            findings=[],
            limitations=[],
            extraction_confidence="low",
        ),
    }

    matrix_rows, candidate_gaps = build_experimental_matrix(records)
    # Missing-data papers should NOT create "CNN on Camelyon17" or "CNN on PACS" gaps
    gap_labels = [g["gap_label"] for g in candidate_gaps]
    assert "CNN on Camelyon17" not in gap_labels
    assert "CNN on PACS" not in gap_labels
    assert any(r["Status"] == "Unknown (missing data)" for r in matrix_rows)
    # Strategies should not be blindly copied onto every cross-product row
    for r in matrix_rows:
        assert "SWAD" not in r.get("Linked Setting", "")


def test_honest_audit_critic_gate_and_hypothesis_rules():
    papers = get_sample_corpus()
    res = run_full_pipeline(papers, llm_fn=None, mode_label="Rule-Based Analysis")

    # Audit must honestly report prompt_used=False and version="not used"
    assert all(not entry.prompt_used for entry in res.audit_log)
    assert all(entry.prompt_version == "not used" for entry in res.audit_log)

    # Critic must NOT accept when prior_status="not run"
    for _, crit in res.critic_reviews:
        assert crit.verdict == "revise"

    # Hypotheses must set expected_direction="unknown" when no quoted passage supports direction
    for _, hyp in res.hypotheses:
        if hyp.decision == "hypothesis":
            assert hyp.expected_direction == "unknown"

    # Corpus signature detects staleness
    sig1 = compute_corpus_signature(papers, "Rule-Based Analysis")
    sig2 = compute_corpus_signature(papers[:2], "Rule-Based Analysis")
    assert sig1 != sig2


def test_experiment_plan_grounded_splits_and_unknown_compute():
    plan = RuleBasedExperimentDesigner.design(
        hypothesis="If CNN is evaluated on Camelyon17, then Accuracy may differ.",
        draft_plan="Test CNN on Camelyon17",
        grounded_options="Datasets: Camelyon17. Baselines: ResNet-50 ERM.",
    )
    assert "hospital-based" in plan.dataset_and_split.lower()
    assert "unknown" in plan.compute_estimate.lower()
    assert "low statistical power" in plan.statistics.lower()
    assert "ERM baseline" not in plan.independent_variable
    assert "Initialize baseline ERM baseline" not in " ".join(plan.procedure)


def test_ask_the_papers_answers_gap_and_unrelated_questions_properly():
    papers = get_sample_corpus()
    res = run_full_pipeline(papers, llm_fn=None)

    # 1. "tell me about the gaps" lists the candidate gaps and ratings
    ans_about, _, _ = answer_workspace_question("tell me about the gaps", res, llm_fn=None)
    assert "Found" in ans_about and "candidate gaps" in ans_about

    # 2. "tell in a simple way so i can understand the gaps so i can build it" gives plain-English explanation + build
    ans_simple, _, _ = answer_workspace_question("tell in a simple way so i can understand the gaps so i can build it", res, llm_fn=None)
    assert "In simple terms:" in ans_simple
    assert "How you can build & test it:" in ans_simple

    # 3. "how do i build with any of the gaps" gives concrete experiment steps
    ans_build, _, _ = answer_workspace_question("how do i build with any of the gaps", res, llm_fn=None)
    assert "Recommended build" in ans_build

    # All 3 answers must be distinct!
    assert ans_about != ans_simple
    assert ans_simple != ans_build
    assert ans_about != ans_build

    # 4. Unrelated question declines instead of matching generic words like "build"
    ans_unrel, passages_unrel, _ = answer_workspace_question("how do i build a rocket ship", res, llm_fn=None)
    assert passages_unrel == ""
    assert "do not answer this question" in ans_unrel.lower()
