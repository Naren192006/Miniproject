"""
Tests for the Innovative Upgrades:
1. Evaluation Harness (16-Paper Grounding Benchmark, Gap Precision, Contradiction Precision, Ablations)
2. Contradiction Confidence Scores & Conflict Factor Attribution
3. Topic & Scope Advisor (grounded in Gap Matrix, mini/major labels, compute, risks)
4. Executable Experiment Runner (script generation, execution, paired t-test)
"""

from hypothesisforge.experiment_runner import execute_experiment_plan, generate_experiment_script
from hypothesisforge.grounding_benchmark import (
    HAND_LABELED_BENCHMARK,
    run_full_evaluation_harness,
    run_grounding_benchmark,
)
from hypothesisforge.pipeline import answer_workspace_question, get_sample_corpus, run_full_pipeline
from hypothesisforge.prompt_engine.fallbacks import (
    RuleBasedContradictionEvaluator,
    RuleBasedExperimentDesigner,
    RuleBasedQueryRewriter,
)
from hypothesisforge.prompt_engine.runner import PromptRunner
from hypothesisforge.scope_advisor import ScopeInterview, recommend_scoped_topics
from hypothesisforge.scope_advisor import recommend_scoped_topics


def test_executable_experiment_runner_end_to_end():
    plan = RuleBasedExperimentDesigner.design(
        hypothesis="If MLP is evaluated on Digits, then Accuracy may differ.",
        draft_plan="Compare MLP vs LogisticRegression",
        grounded_options="Datasets: PACS. Baselines: ResNet-50 ERM.",
    )
    script = generate_experiment_script(plan, n_seeds=3)
    assert "load_digits" in script
    assert "ttest_rel" in script

    exec_res = execute_experiment_plan(plan, n_seeds=3)
    assert len(exec_res.seeds) == 3
    assert len(exec_res.baseline_scores) == 3
    assert len(exec_res.proposed_scores) == 3
    assert exec_res.outcome in {"confirmed", "refuted", "inconclusive"}
    assert 0.0 <= exec_res.p_value <= 1.0


def test_grounding_benchmark_16_papers_and_full_harness():
    assert len(HAND_LABELED_BENCHMARK) == 16
    report = run_full_evaluation_harness()
    assert len(report.extraction_rows) == 3

    plain, ab_noise, hf = report.extraction_rows
    assert hf.unsupported_claim_rate < plain.unsupported_claim_rate
    assert hf.unsupported_claim_rate == 0.0
    assert hf.quote_pass_rate == 100.0
    assert hf.precision == 100.0
    # Ablation must have higher unsupported-claim rate than full HypothesisForge
    assert ab_noise.unsupported_claim_rate > hf.unsupported_claim_rate
    assert report.gap_precision == 100.0
    assert report.contradiction_precision == 100.0


def test_contradiction_confidence_and_conflict_factors():
    pair_diff_ds = (
        "Side A:\n- Paper: P01\n- Value: 84.6\n- Metric: Accuracy\n- Dataset: PACS\n"
        "- Setting: ResNet-50\n- Evaluation protocol: 3 seeds\n\n"
        "Side B:\n- Paper: P02\n- Value: 66.8\n- Metric: Accuracy\n- Dataset: OfficeHome\n"
        "- Setting: ViT\n- Evaluation protocol: 3 seeds"
    )
    out = RuleBasedContradictionEvaluator.evaluate(pair_diff_ds)
    assert out.classification == "non-comparable evidence"
    assert out.confidence_score >= 0.80
    assert "dataset" in out.conflict_factors
    assert "hyperparameters" in out.conflict_factors

    pair_missing = (
        "Side A:\n- Paper: P01\n- Value: 80.0\n- Metric: Accuracy\n- Dataset: Unknown\n"
        "- Setting: not stated\n- Evaluation protocol: not stated\n\n"
        "Side B:\n- Paper: P02\n- Value: 75.0\n- Metric: Accuracy\n- Dataset: PACS\n"
        "- Setting: ResNet-50\n- Evaluation protocol: 3 seeds"
    )
    out2 = RuleBasedContradictionEvaluator.evaluate(pair_missing)
    assert out2.classification == "insufficient information"
    assert out2.confidence_score <= 0.50
    assert "split" in out2.conflict_factors


def test_scope_advisor_grounded_in_gap_matrix():
    papers = get_sample_corpus()
    res = run_full_pipeline(papers, llm_fn=None)
    interview = ScopeInterview(
        team_size=2,
        timeline_weeks=4,
        compute_budget="CPU / Colab Free",
        goal="Mini Project",
    )
    recs = recommend_scoped_topics(interview, res, max_options=5)
    assert len(recs) >= 1
    gap_labels = {g["gap_label"] for g in res.candidate_gaps}
    for rec in recs:
        assert rec.gap_label in gap_labels
        assert rec.scope_label in {"mini", "major"}
        assert len(rec.motivating_papers) >= 1
        assert len(rec.risks) >= 1


def test_prompt_09_query_rewriter_and_pronoun_resolution():
    # Pronoun resolution across chat history
    rw = RuleBasedQueryRewriter.rewrite(
        history="user: What accuracy does ViT-B/16 achieve on PACS?",
        question="What about ResNet-50 on that dataset?",
    )
    assert "PACS" in rw.standalone_question
    assert rw.intent == "factual_qa"
    assert len(rw.sub_queries) <= 3

    # Multi-part splitting + intent classification
    rw_gap = RuleBasedQueryRewriter.rewrite("", "What are the gaps and how do I build it?")
    assert rw_gap.intent == "gap"
    assert len(rw_gap.sub_queries) == 2

    # PromptRunner integration
    res = PromptRunner().run_query_rewrite(
        history="user: What about PACS?",
        question="Compare ResNet-50 and ViT on it",
        llm_fn=None,
    )
    assert res.data.intent == "compare"
    assert "PACS" in res.data.standalone_question

    # End-to-end follow-up in answer_workspace_question
    pipeline_res = run_full_pipeline(get_sample_corpus(), llm_fn=None)
    ans, _, note = answer_workspace_question(
        question="What accuracy does ResNet-50 achieve on it?",
        result=pipeline_res,
        llm_fn=None,
        chat_history=[{"role": "user", "content": "Tell me about PACS"}],
    )
    assert "84.6%" in ans or "80.2%" in ans


def test_topic_and_scope_advisor_grounded_in_gap_matrix():
    papers = get_sample_corpus()
    res = run_full_pipeline(papers, llm_fn=None)
    interview = ScopeInterview(team_size=2, timeline_weeks=6, compute_budget="CPU / Colab Free", goal="Mini Project")
    options = recommend_scoped_topics(interview, res, max_options=5)
    assert len(options) >= 1
    assert all(opt.scope_label in {"mini", "major"} for opt in options)
    assert all(len(opt.motivating_papers) >= 1 for opt in options)
    assert all(len(opt.risks) >= 1 for opt in options)
    # Every recommended option must come directly from the gap matrix
    gap_labels = {g["gap_label"] for g in res.candidate_gaps}
    assert all(opt.gap_label in gap_labels for opt in options)


def test_prompt_10_topic_interviewer():
    runner = PromptRunner()
    # Empty answers -> asks first missing field
    r1 = runner.run_topic_interview(answers={}, llm_fn=None)
    assert r1.data.ready is False
    assert "field_of_interest" in r1.data.missing_fields
    assert len(r1.data.next_question) >= 5

    # Partial answers -> skips answered fields
    r2 = runner.run_topic_interview(
        answers={"field_of_interest": "Domain Generalization", "project_level": "mini"},
        llm_fn=None,
    )
    assert r2.data.ready is False
    assert "field_of_interest" not in r2.data.missing_fields
    assert "project_level" not in r2.data.missing_fields

    # All 6 required fields present -> ready=True
    full_answers = {
        "field_of_interest": "Domain Generalization",
        "current_skills": "Python, PyTorch",
        "available_weeks": "6",
        "compute_access": "laptop",
        "project_level": "mini",
        "preferred_outcome": "working prototype",
    }
    r3 = runner.run_topic_interview(answers=full_answers, llm_fn=None)
    assert r3.data.ready is True
    assert r3.data.missing_fields == []


def test_prompt_11_topic_recommender():
    from hypothesisforge.prompt_engine.validator import ResponseValidator

    runner = PromptRunner()
    res = run_full_pipeline(get_sample_corpus(), llm_fn=None)
    gaps = [
        {
            "gap_label": g["gap_label"],
            "architecture": g["architecture"],
            "dataset": g["dataset"],
            "supporting_paper_ids": [p.paper_id for p in res.papers[:1]],
        }
        for g in res.candidate_gaps[:4]
    ]
    out = runner.run_topic_recommendation(
        profile={"available_weeks": 6, "compute_access": "laptop", "project_level": "mini"},
        gaps=gaps,
        contradictions=[],
        scope_criteria="mini",
        n_papers=len(res.papers),
        llm_fn=None,
    )
    assert 1 <= len(out.data.options) <= 5
    for opt in out.data.options:
        assert opt.scope == "mini"
        assert f"not found in the {len(res.papers)} papers searched" in opt.reason.lower()
        assert len(opt.evidence_ids) >= 1

    # Validator rejects forbidden words like "novel" or "state of the art"
    bad_json = (
        '{"options":[{"title":"A novel study","scope":"mini","reason":"It is the first and unexplored.",'
        '"evidence_ids":["P-1"],"datasets":["PACS"],"compute_needed":"Laptop","papers_to_read":["P-1"],'
        '"risks":["Small data"],"effort_weeks":4}],"warnings":["Fewer than 3 options."]}'
    )
    val = ResponseValidator.validate_topic_recommendation(bad_json)
    assert val.is_valid is False


def test_prompt_12_conflict_explainer():
    from hypothesisforge.prompt_engine.validator import ResponseValidator

    runner = PromptRunner()

    # 1. True conflict when dataset, metric, and split all match and values differ
    rec_a = {
        "dataset": "PACS",
        "metric": "Accuracy",
        "split": "Leave-one-domain-out",
        "architecture": "ResNet-50",
        "hyperparameters": "AdamW lr=1e-4",
        "value": "84.6%",
        "verbatim_quote": "ResNet-50 achieves 84.6% accuracy on PACS.",
        "page": 4,
    }
    rec_b = {
        "dataset": "PACS",
        "metric": "Accuracy",
        "split": "Leave-one-domain-out",
        "architecture": "ResNet-50",
        "hyperparameters": "AdamW lr=1e-4",
        "value": "78.1%",
        "verbatim_quote": "ResNet-50 reaches 78.1% accuracy on PACS.",
        "page": 6,
    }
    res_true = runner.run_conflict_explainer(rec_a, rec_b, llm_fn=None)
    assert res_true.data.verdict == "true_conflict"
    assert res_true.data.confidence == "high"
    assert res_true.data.page_a == 4 and res_true.data.page_b == 6

    # 2. Condition mismatch when dataset or split differs
    rec_c = dict(rec_b, dataset="OfficeHome", split="Random 80/20")
    res_mis = runner.run_conflict_explainer(rec_a, rec_c, llm_fn=None)
    assert res_mis.data.verdict == "condition_mismatch"
    assert "dataset" in res_mis.data.differing_factors
    assert "split" in res_mis.data.differing_factors

    # 3. Insufficient info when split is missing
    rec_missing = dict(rec_b, split="")
    res_insuf = runner.run_conflict_explainer(rec_a, rec_missing, llm_fn=None)
    assert res_insuf.data.verdict == "insufficient_info"

    # 4. Validator rejects quotes longer than 25 words
    long_q = " ".join(["word"] * 26)
    bad_json = (
        f'{{"verdict":"true_conflict","differing_factors":[],"quote_a":"{long_q}","page_a":1,'
        '"quote_b":"Short quote","page_b":2,"confidence":"high","explanation":"Both papers match on conditions."}'
    )
    assert ResponseValidator.validate_conflict_explainer(bad_json).is_valid is False


def test_prompt_13_prior_work_evidence():
    from hypothesisforge.prompt_engine.validator import ResponseValidator

    runner = PromptRunner()
    hyp = "If ResNet-50 is evaluated on PACS, then Accuracy changes relative to ViT-B/16."

    # 1. Partial overlap on architecture + dataset
    search_results = [
        {
            "id": "AX-1",
            "title": "Domain Generalization with ResNet-50 on PACS",
            "year": 2023,
            "abstract": "We evaluate ResNet-50 on PACS and report accuracy across four domains.",
        },
        {
            "id": "AX-2",
            "title": "Unrelated NLP Study",
            "year": 2022,
            "abstract": "Language modeling on WikiText using transformers.",
        },
    ]
    res = runner.run_prior_work_evidence(hyp, search_results, llm_fn=None)
    assert len(res.data.closest) <= 5
    assert res.data.closest[0].id == "AX-1"
    assert "architecture" in res.data.closest[0].overlap
    assert "dataset" in res.data.closest[0].overlap

    # 2. No overlap -> exact required phrasing
    no_overlap_results = [
        {"id": "AX-9", "title": "Tabular XGBoost", "year": 2021, "abstract": "Gradient boosting on census data."}
    ]
    res_none = runner.run_prior_work_evidence(hyp, no_overlap_results, llm_fn=None)
    assert res_none.data.statement == "No close match among 1 results searched; manual check required."

    # 3. Validator rejects novelty claims
    bad_json = '{"closest":[],"statement":"This hypothesis is completely novel."}'
    assert ResponseValidator.validate_prior_work_evidence(bad_json).is_valid is False


def test_prompt_14_spoken_explainer():
    from hypothesisforge.prompt_engine.models import count_words
    from hypothesisforge.prompt_engine.validator import ResponseValidator

    runner = PromptRunner()
    payload = {
        "title": "ResNet-50 on OfficeHome",
        "verdict": "condition_mismatch",
        "explanation": "This finding shows two papers used different splits.",
        "evidence_ids": ["P-101"],
        "sources": ["P-101 p.4", "P-202 p.7"],
    }
    res = runner.run_spoken_explanation(payload, llm_fn=None)
    assert count_words(res.data.spoken_text) <= 120
    assert "finding" not in res.data.spoken_text.lower()
    assert "candidate" in res.data.spoken_text.lower()
    assert "P-101 p.4" in res.data.spoken_text
    assert "P-202 p.7" in res.data.spoken_text
    assert res.data.sources == ["P-101 p.4", "P-202 p.7"]

    # Validator rejects 'finding' and > 120 words
    bad_finding = '{"spoken_text":"This finding is important. Check P-1 p.1 on screen.","sources":["P-1 p.1"]}'
    assert ResponseValidator.validate_spoken_explanation(bad_finding).is_valid is False

    too_long = " ".join(["word"] * 125)
    bad_len = f'{{"spoken_text":"{too_long}","sources":["P-1 p.1"]}}'
    assert ResponseValidator.validate_spoken_explanation(bad_len).is_valid is False


def test_prompt_15_grounded_qa():
    from hypothesisforge.prompt_engine.models import count_words
    from hypothesisforge.prompt_engine.validator import ResponseValidator

    runner = PromptRunner()
    passages = [
        {"id": "P-101", "page": 4, "text": "ResNet-50 achieves 84.6% accuracy on PACS under leave-one-domain-out."},
        {"id": "P-102", "page": 7, "text": "ViT-B/16 reaches 88.2% accuracy on OfficeHome."},
    ]

    # 1. Answerable question
    res_ok = runner.run_grounded_qa("What accuracy does ResNet-50 achieve on PACS?", passages, llm_fn=None)
    assert res_ok.data.answerable is True
    assert len(res_ok.data.claims) >= 1
    assert res_ok.data.claims[0].paper_id == "P-101"
    assert res_ok.data.claims[0].page == 4
    assert count_words(res_ok.data.claims[0].quote) <= 25

    # 2. Unanswerable question -> answerable=False
    res_no = runner.run_grounded_qa("What is the boiling point of liquid nitrogen?", passages, llm_fn=None)
    assert res_no.data.answerable is False
    assert "missing" in res_no.data.answer.lower()

    # 3. Validator rejects > 25-word quote or ungrounded quote
    long_q = " ".join(["word"] * 26)
    bad_len = f'{{"answerable":true,"answer":"Answer [P-101, 4].","claims":[{{"text":"Claim text","paper_id":"P-101","page":4,"quote":"{long_q}"}}],"follow_ups":[]}}'
    assert ResponseValidator.validate_grounded_qa(bad_len).is_valid is False

    bad_ungrounded = '{"answerable":true,"answer":"Answer [P-101, 4].","claims":[{"text":"Claim text","paper_id":"P-101","page":4,"quote":"Fabricated quote not in text."}],"follow_ups":[]}'
    assert ResponseValidator.validate_grounded_qa(bad_ungrounded, passages_source=passages[0]["text"]).is_valid is False


def test_ask_papers_what_can_i_build_not_out_of_scope():
    pipeline_res = run_full_pipeline(get_sample_corpus(), llm_fn=None)
    ans, _, note = answer_workspace_question(
        question="what can i build? and what should i use to build it",
        result=pipeline_res,
        llm_fn=None,
    )
    assert "out_of_scope" not in note
    assert "What you can build" in ans
    assert "What to use to build it" in ans

    ans_road, _, note_road = answer_workspace_question(
        question="tell me the roadmap",
        result=pipeline_res,
        llm_fn=None,
    )
    assert "topic_advice" in note_road
    assert "Project Roadmap" in ans_road


def test_rag_typo_tolerance_boilerplate_filtering_and_tfidf():
    from hypothesisforge.ingestion import TextChunk, retrieve_passages
    from hypothesisforge.prompt_engine.fallbacks import RuleBasedEvidenceAnswerer

    # 1. Typo-tolerant summary routing ("summaarize the researchh papers")
    pipeline_res = run_full_pipeline(get_sample_corpus(), llm_fn=None)
    ans_sum, _, note_sum = answer_workspace_question(
        question="summaarize the researchh papers",
        result=pipeline_res,
        llm_fn=None,
    )
    assert "summary" in note_sum
    assert len(ans_sum) > 20

    # 2. Boilerplate filtering + TF-IDF ranking
    chunks = [
        TextChunk(1, "P-1", "Paper 1", 10, "Acknowledgements Funded by DFG. Author Contributions and Competing Interests."),
        TextChunk(2, "P-1", "Paper 1", 2, "DenseNet-121 achieved 92.4% AUC on ChestX-ray14 multi-label classification."),
    ]
    formatted, selected = retrieve_passages("what accuracy did DenseNet get?", chunks, top_k=2)
    assert len(selected) == 1
    assert selected[0].page == 2
    assert "Acknowledgements" not in formatted

    # 3. Synonym/stem matching in RuleBasedEvidenceAnswerer
    ans_ev = RuleBasedEvidenceAnswerer.answer("what accuracy did DenseNet get?", formatted)
    assert ans_ev.is_declined is False
    assert "92.4%" in ans_ev.text
