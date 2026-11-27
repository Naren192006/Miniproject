"""
HypothesisForge - End-to-End Research Analysis Pipeline
Orchestrates Stages 1-8 using PromptRunner, builds the Experimental Matrix,
Knowledge Graph (NetworkX), and records honest prompt version traceability.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx

from .ingestion import (
    PaperDocument,
    TextChunk,
    chunk_pages,
    format_pages_for_prompt,
    make_stable_paper_id,
    retrieve_passages,
)
from .prompt_engine.fallbacks import canonicalize_architecture, is_cnn_architecture
from .prompt_engine.runner import PromptRunner, PromptExecutionResult, LLMCallable
from .prompt_engine.models import (
    LiteratureExtractionOutput,
    GapRelevanceOutput,
    ContradictionJudgementOutput,
    HypothesisGenerationOutput,
    ScientificCriticOutput,
    ExperimentDesignOutput,
    ReportSummaryOutput,
)


def compute_corpus_signature(papers: List[PaperDocument], mode_label: str = "") -> str:
    """Computes a deterministic signature for the active corpus + mode to detect stale reports."""
    payload = "|".join(f"{p.paper_id}:{p.title}:{len(p.pages)}" for p in papers) + f"::{mode_label}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def get_sample_corpus() -> List[PaperDocument]:
    """
    Returns a curated 3-paper Domain Generalization corpus with stable Paper IDs.
    """
    raw_papers = [
        {
            "title": "Domain-Adversarial Invariance for Visual Recognition",
            "year": "2023",
            "pages": [
                (
                    1,
                    "In this paper, we study Domain Generalization for image classification across unseen target domains. "
                    "We evaluate ResNet-50 and ResNet-18 backbones on the PACS dataset under a Leave-one-domain-out protocol."
                ),
                (
                    2,
                    "We compare Empirical Risk Minimization (ERM) against Domain-adversarial training (DANN) using 3 random seeds. "
                    "On PACS, ResNet-50 with Domain-adversarial training achieves 84.6% Accuracy, whereas the ERM baseline reaches 80.2% Accuracy."
                ),
                (
                    3,
                    "Findings: Domain-adversarial training improves generalization on Sketch and Cartoon shifts. "
                    "Limitations: Computational cost increases substantially with the number of source domains, and we did not evaluate Vision Transformer backbones."
                ),
            ],
        },
        {
            "title": "Benchmarking Empirical Risk Minimization on OfficeHome",
            "year": "2023",
            "pages": [
                (
                    1,
                    "We investigate Domain Generalization under Covariate shift on the OfficeHome benchmark. "
                    "All experiments use a ResNet-50 architecture with Leave-one-domain-out splits across Art, Clipart, Product, and Real-World."
                ),
                (
                    2,
                    "Evaluating across 3 random seeds, standard ERM achieves 66.8% Accuracy on OfficeHome, while CORAL reaches 68.4% Accuracy and Mixup achieves 67.9% Accuracy."
                ),
                (
                    3,
                    "Limitations: ResNet-50 struggles on Clipart-to-Real-World transfer when style shifts are severe. Future work should examine transformer architectures."
                ),
            ],
        },
        {
            "title": "Self-Attention Representations Under Out-of-Distribution Shifts",
            "year": "2024",
            "pages": [
                (
                    1,
                    "We evaluate Vision Transformer (ViT-B/16) models on Domain Generalization tasks using the PACS and VLCS benchmarks under Leave-one-domain-out evaluation."
                ),
                (
                    2,
                    "Trained with ERM across 5 random seeds, ViT-B/16 achieves 86.2% Accuracy on PACS and 78.5% Accuracy on VLCS, outperforming CNN baselines."
                ),
                (
                    3,
                    "Limitations: Large pretraining datasets may overlap with photographic target domains, and OfficeHome was not evaluated due to compute constraints."
                ),
            ],
        },
    ]

    docs: List[PaperDocument] = []
    for p in raw_papers:
        paper_id = make_stable_paper_id(p["title"])
        formatted = format_pages_for_prompt(p["pages"])
        chunks = chunk_pages(paper_id=paper_id, title=p["title"], pages=p["pages"])
        docs.append(
            PaperDocument(
                paper_id=paper_id,
                title=p["title"],
                year=p["year"],
                pages=p["pages"],
                formatted_text=formatted,
                extraction_status="ok",
                chunks=chunks,
            )
        )
    return docs


@dataclass
class AuditEntry:
    stage: str
    item_id: str
    prompt_name: str
    prompt_version: str
    prompt_used: bool
    retries_used: int
    used_fallback: bool
    validation_error: Optional[str] = None


@dataclass
class AnalysisPipelineResult:
    corpus_signature: str
    papers: List[PaperDocument]
    all_chunks: List[TextChunk]
    literature_records: Dict[str, LiteratureExtractionOutput]
    matrix_rows: List[Dict[str, Any]]
    candidate_gaps: List[Dict[str, Any]]
    gap_assessments: List[Tuple[str, GapRelevanceOutput]]
    contradictions: List[Tuple[str, ContradictionJudgementOutput]]
    hypotheses: List[Tuple[str, HypothesisGenerationOutput]]
    critic_reviews: List[Tuple[str, ScientificCriticOutput]]
    experiment_plans: List[Tuple[str, ExperimentDesignOutput]]
    report_summary: Optional[ReportSummaryOutput]
    knowledge_graph: nx.DiGraph
    audit_log: List[AuditEntry] = field(default_factory=list)


def _record_audit(audit_log: List[AuditEntry], stage: str, item_id: str, res: PromptExecutionResult) -> None:
    audit_log.append(
        AuditEntry(
            stage=stage,
            item_id=item_id,
            prompt_name=res.prompt_name,
            prompt_version=res.prompt_version,
            prompt_used=res.prompt_used,
            retries_used=res.retries_used,
            used_fallback=res.used_fallback,
            validation_error=res.validation_error,
        )
    )


def build_experimental_matrix(
    records: Dict[str, LiteratureExtractionOutput]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compares Architecture x Dataset combinations ONLY when both values are explicitly reported.
    Missing values are marked 'Unknown (incomplete)' and excluded from candidate-gap generation.
    Strategies are not blindly copied across cross-products unless linked in a reported result.
    """
    observed_pairs = set()
    complete_archs = set()
    complete_datasets = set()
    matrix_rows: List[Dict[str, Any]] = []

    for pid, rec in records.items():
        archs = [canonicalize_architecture(a) for a in rec.architectures] if rec.architectures else []
        datasets = list(dict.fromkeys(rec.datasets)) if rec.datasets else []

        # Map reported results that explicitly link an architecture+dataset to a setting
        linked_settings: Dict[Tuple[str, str], str] = {}
        for r in rec.reported_results:
            if r.dataset not in {"not stated", "Unknown"} and r.setting != "not stated":
                for a in archs:
                    if a.lower() in r.setting.lower() or r.setting.lower() in a.lower():
                        linked_settings[(a, r.dataset)] = r.setting

        if not archs or not datasets:
            matrix_rows.append(
                {
                    "Paper ID": pid,
                    "Architecture": ", ".join(archs) if archs else "Unknown (not stated)",
                    "Dataset": ", ".join(datasets) if datasets else "Unknown (not stated)",
                    "Linked Setting": "Incomplete record (missing architecture or dataset)",
                    "Status": "Unknown (missing data)",
                }
            )
            continue

        for a in dict.fromkeys(archs):
            complete_archs.add(a)
            for d in datasets:
                complete_datasets.add(d)
                observed_pairs.add((a, d))
                setting_note = linked_settings.get((a, d), "Not linked per pair (check source)")
                matrix_rows.append(
                    {
                        "Paper ID": pid,
                        "Architecture": a,
                        "Dataset": d,
                        "Linked Setting": setting_note,
                        "Status": "Observed in corpus",
                    }
                )

    candidate_gaps: List[Dict[str, Any]] = []
    for a in sorted(complete_archs):
        for d in sorted(complete_datasets):
            if (a, d) not in observed_pairs:
                # Avoid redundant "CNN on X" gap if a specific CNN (like ResNet-50) is already observed on X
                if a.lower() == "cnn" and any(is_cnn_architecture(obs_a) and obs_d == d for obs_a, obs_d in observed_pairs):
                    continue
                candidate_gaps.append(
                    {
                        "gap_label": f"{a} on {d}",
                        "architecture": a,
                        "dataset": d,
                    }
                )
                matrix_rows.append(
                    {
                        "Paper ID": "—",
                        "Architecture": a,
                        "Dataset": d,
                        "Linked Setting": "—",
                        "Status": "Candidate gap (unobserved)",
                    }
                )

    return matrix_rows, candidate_gaps


def build_knowledge_graph(
    records: Dict[str, LiteratureExtractionOutput],
    hypotheses: List[Tuple[str, HypothesisGenerationOutput]],
) -> nx.DiGraph:
    """Builds a NetworkX Knowledge Graph with canonicalized architecture labels."""
    G = nx.DiGraph()

    for pid, rec in records.items():
        G.add_node(pid, node_type="Paper", label=pid)
        for arch in rec.architectures:
            canon_arch = canonicalize_architecture(arch)
            G.add_node(canon_arch, node_type="Architecture", label=canon_arch)
            G.add_edge(pid, canon_arch, relation="USES_ARCHITECTURE")
        for ds in rec.datasets:
            G.add_node(ds, node_type="Dataset", label=ds)
            G.add_edge(pid, ds, relation="EVALUATED_ON")
        for strat in rec.training_strategies:
            G.add_node(strat, node_type="Method", label=strat)
            G.add_edge(pid, strat, relation="USES_METHOD")

    for idx, (gap_label, hyp) in enumerate(hypotheses, start=1):
        if hyp.decision == "hypothesis":
            h_id = f"H{idx}"
            G.add_node(h_id, node_type="Hypothesis", label=h_id)
            for pid in hyp.cited_paper_ids or []:
                if pid in G:
                    G.add_edge(pid, h_id, relation="MOTIVATES")

    return G


def run_full_pipeline(
    papers: List[PaperDocument],
    llm_fn: Optional[LLMCallable] = None,
    mode_label: str = "Rule-Based Analysis",
    topic: str = "",
) -> AnalysisPipelineResult:
    """Executes all HypothesisForge stages end-to-end and returns structured results."""
    runner = PromptRunner()
    audit_log: List[AuditEntry] = []
    corpus_sig = compute_corpus_signature(papers, mode_label)

    all_chunks: List[TextChunk] = []
    for p in papers:
        all_chunks.extend(p.chunks)

    # Stage 1: Literature Extraction (Prompt 01)
    literature_records: Dict[str, LiteratureExtractionOutput] = {}
    for p in papers:
        res = runner.run_literature_extraction(
            title=p.title,
            year=p.year,
            paper_text=p.formatted_text,
            llm_fn=llm_fn,
        )
        _record_audit(audit_log, "literature", p.paper_id, res)
        literature_records[p.paper_id] = res.data  # type: ignore

    # Experimental Matrix & Candidate Gaps (handles missing data properly)
    matrix_rows, candidate_gaps = build_experimental_matrix(literature_records)

    related_lines = []
    for pid, rec in literature_records.items():
        canon_archs = sorted({canonicalize_architecture(a) for a in rec.architectures})
        related_lines.append(
            f"- {pid}: Architectures={canon_archs}, Datasets={rec.datasets}, "
            f"Strategies={rec.training_strategies}, Shifts={rec.shift_settings}"
        )
    related_records_str = "\n".join(related_lines)

    # Stage 2: Gap Relevance (Prompt 02)
    gap_assessments: List[Tuple[str, GapRelevanceOutput]] = []
    for gap_info in candidate_gaps[:4]:
        gap_label = gap_info["gap_label"]
        res = runner.run_gap_relevance(
            gap=gap_label,
            related_records=related_records_str,
            llm_fn=llm_fn,
        )
        _record_audit(audit_log, "gaps", gap_label, res)
        gap_assessments.append((gap_label, res.data))  # type: ignore

    # Stage 3: Contradiction Judgement (Prompt 03)
    contradictions: List[Tuple[str, ContradictionJudgementOutput]] = []
    result_entries: List[Tuple[str, LiteratureExtractionOutput, Any]] = []
    for pid, rec in literature_records.items():
        if rec.reported_results:
            result_entries.append((pid, rec, rec.reported_results[0]))

    if len(result_entries) >= 2:
        pairs_checked = 0
        for i in range(len(result_entries)):
            for j in range(i + 1, len(result_entries)):
                if pairs_checked >= 6:
                    break
                pid_a, rec_a, r_a = result_entries[i]
                pid_b, rec_b, r_b = result_entries[j]
                pair_str = (
                    f"Side A:\n- Paper: {pid_a}\n- Value: {r_a.value}\n- Metric: {r_a.metric}\n"
                    f"- Dataset: {r_a.dataset}\n- Setting: {r_a.setting}\n"
                    f"- Evaluation protocol: {rec_a.evaluation_protocol}\n- Quote: \"{r_a.quote}\"\n\n"
                    f"Side B:\n- Paper: {pid_b}\n- Value: {r_b.value}\n- Metric: {r_b.metric}\n"
                    f"- Dataset: {r_b.dataset}\n- Setting: {r_b.setting}\n"
                    f"- Evaluation protocol: {rec_b.evaluation_protocol}\n- Quote: \"{r_b.quote}\""
                )
                pair_title = f"{pid_a} ({r_a.dataset}) vs {pid_b} ({r_b.dataset})"
                res = runner.run_contradiction_judgement(pair=pair_str, llm_fn=llm_fn)
                _record_audit(audit_log, "contradictions", pair_title, res)
                contradictions.append((pair_title, res.data))  # type: ignore
                pairs_checked += 1
            if pairs_checked >= 6:
                break

    # Stage 4: Hypothesis Generation (Prompt 04)
    hypotheses: List[Tuple[str, HypothesisGenerationOutput]] = []
    for gap_info in candidate_gaps[:3]:
        gap_label = gap_info["gap_label"]
        gap_with_evidence = f"Candidate gap: {gap_label}\nEvidence:\n{related_records_str}"
        res = runner.run_hypothesis_generation(gap=gap_with_evidence, llm_fn=llm_fn)
        _record_audit(audit_log, "hypotheses", gap_label, res)
        hypotheses.append((gap_label, res.data))  # type: ignore

    # Stage 5: Scientific Critic (Prompt 05) — honestly pass prior_status="not run" since external search didn't run
    critic_reviews: List[Tuple[str, ScientificCriticOutput]] = []
    for idx, (gap_label, hyp) in enumerate(hypotheses, start=1):
        if hyp.decision != "hypothesis":
            continue
        hyp_text = hyp.statement or gap_label
        checks_str = f"Traceability: passed. Cited papers: {hyp.cited_paper_ids}. Baseline: {hyp.baseline}."
        res = runner.run_scientific_critic(
            hypothesis=hyp_text,
            checks=checks_str,
            prior_status="not run",
            overlap="Not searched",
            llm_fn=llm_fn,
        )
        _record_audit(audit_log, "critic", f"H{idx}", res)
        critic_reviews.append((hyp_text, res.data))  # type: ignore

    # Stage 6: Experiment Design (Prompt 06)
    grounded_datasets = sorted({d for rec in literature_records.values() for d in rec.datasets})
    grounded_strats = sorted({s for rec in literature_records.values() for s in rec.training_strategies})
    grounded_options = f"Datasets: {', '.join(grounded_datasets)}. Baselines: {', '.join(grounded_strats)}."

    experiment_plans: List[Tuple[str, ExperimentDesignOutput]] = []
    for idx, (hyp_text, critique) in enumerate(critic_reviews, start=1):
        if critique.verdict == "reject":
            continue
        res = runner.run_experiment_design(
            hypothesis=hyp_text,
            draft_plan=f"Test {hyp_text} using grounded baselines.",
            grounded_options=grounded_options,
            llm_fn=llm_fn,
        )
        _record_audit(audit_log, "experiments", f"E{idx}", res)
        experiment_plans.append((hyp_text, res.data))  # type: ignore

    # Stage 7: Report Summary (Prompt 07)
    declined_count = sum(1 for _, h in hypotheses if h.decision == "insufficient_evidence")
    protocols_stated_count = sum(
        1 for rec in literature_records.values()
        if rec.evaluation_protocol and rec.evaluation_protocol.lower() != "not stated"
    )
    unrun_list = ["external prior-work search"]
    if llm_fn is None:
        unrun_list.append("LLM semantic extraction")

    facts_payload = json.dumps(
        {
            "topic": topic or "the examined papers",
            "paper_count": len(papers),
            "datasets": grounded_datasets,
            "protocols_stated_count": protocols_stated_count,
            "candidate_gaps_count": len(candidate_gaps),
            "contradictions_checked": len(contradictions),
            "hypotheses_accepted": sum(1 for _, c in critic_reviews if c.verdict == "accept"),
            "hypotheses_declined": declined_count,
            "unrun_analyses": unrun_list,
        }
    )
    report_res = runner.run_report_summary(facts=facts_payload, llm_fn=llm_fn)
    _record_audit(audit_log, "report", "run_summary", report_res)
    report_summary: ReportSummaryOutput = report_res.data  # type: ignore

    kg = build_knowledge_graph(literature_records, hypotheses)

    return AnalysisPipelineResult(
        corpus_signature=corpus_sig,
        papers=papers,
        all_chunks=all_chunks,
        literature_records=literature_records,
        matrix_rows=matrix_rows,
        candidate_gaps=candidate_gaps,
        gap_assessments=gap_assessments,
        contradictions=contradictions,
        hypotheses=hypotheses,
        critic_reviews=critic_reviews,
        experiment_plans=experiment_plans,
        report_summary=report_summary,
        knowledge_graph=kg,
        audit_log=audit_log,
    )


def answer_workspace_question(
    question: str,
    result: AnalysisPipelineResult,
    llm_fn: Optional[LLMCallable] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str, str]:
    """
    Answers user questions in Ask the Papers using Prompt 09 (Query Rewriter & Intent Router)
    followed by either structured analysis routing or Prompt 08 passage Q&A.
    Returns (answer_text, retrieved_passages_text, mode_note).
    """
    runner = PromptRunner()
    history_str = ""
    if chat_history:
        recent = chat_history[-4:]
        history_str = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')[:160]}" for m in recent)

    rewrite_res = runner.run_query_rewrite(history=history_str, question=question, llm_fn=llm_fn)
    rw = rewrite_res.data
    standalone_q = rw.standalone_question
    intent = rw.intent
    q_low = standalone_q.lower()

    if intent == "out_of_scope":
        return (
            "The passages do not answer this question. It appears outside the scope of the loaded research papers.",
            "",
            f"Intent: out_of_scope · Rewritten: \"{standalone_q}\"",
        )

    if intent == "summary":
        summary_text = result.report_summary.text if result.report_summary else "No summary available yet."
        return summary_text, "", f"Intent: summary · Rewritten: \"{standalone_q}\""

    if intent in {"gap", "contradiction", "compare", "topic_advice"} or any(
        kw in q_low for kw in ["hypothesis", "hypotheses", "critic", "experiment"]
    ):
        all_archs = sorted({a for rec in result.literature_records.values() for a in rec.architectures})
        all_datasets = sorted({d for rec in result.literature_records.values() for d in rec.datasets})
        all_methods = sorted({m for rec in result.literature_records.values() for m in rec.training_strategies})

        if not result.candidate_gaps and not result.experiment_plans and not result.contradictions and not all_archs and not all_datasets:
            return (
                "No candidate gaps, contradictions, models, or datasets were extracted from the current corpus yet. "
                "Try searching a broader topic or uploading papers with explicit models and benchmarks.",
                "",
                f"Intent: {intent} (No data)",
            )

        wants_simple = any(w in q_low for w in ["simple", "understand", "explain", "easy", "plain", "beginner"])
        wants_build = any(w in q_low for w in ["build", "experiment", "steps", "run", "implement", "how do i", "what should i use"])
        wants_hypotheses = any(w in q_low for w in ["hypothesis", "hypotheses", "critic"])

        lines: List[str] = []

        if intent == "contradiction" or intent == "compare":
            if not result.contradictions:
                lines.append("No cross-paper contradictions could be checked yet because fewer than two papers reported comparable numeric metrics.")
            else:
                lines.append(f"Checked {len(result.contradictions)} cross-paper result pairs:")
                for pair_title, cj in result.contradictions[:4]:
                    factors_txt = f" [factors: {', '.join(cj.conflict_factors)}]" if getattr(cj, "conflict_factors", None) else ""
                    lines.append(f"- **{pair_title}** ({cj.classification}{factors_txt}): {cj.explanation}")
        elif wants_hypotheses:
            if not result.hypotheses:
                lines.append("No hypotheses were generated from the current corpus.")
            else:
                lines.append(f"Generated {len(result.hypotheses)} testable hypotheses:")
                for idx, (gap_label, hyp) in enumerate(result.hypotheses[:4], start=1):
                    lines.append(f"- **H{idx} ({gap_label}):** {hyp.statement or hyp.reason}")
        elif intent == "topic_advice":
            wants_roadmap = any(w in q_low for w in ["roadmap", "step by step", "step-by-step", "how to start", "action plan", "schedule"])
            if wants_roadmap:
                top_ds = ", ".join(all_datasets[:2]) or "a shared benchmark dataset"
                top_models = ", ".join(all_archs[:4]) or "your selected baseline and candidate models"
                top_methods = ", ".join(all_methods[:3]) or "standard training & fine-tuning"
                top_papers = ", ".join(p.paper_id for p in result.papers[:3])
                lines.append(f"**Project Roadmap (grounded in {len(result.papers)} loaded papers):**")
                lines.append(f"1. **Read & Lock Setup:** Review {top_papers} and fix one shared dataset ({top_ds}) and train/test split.")
                lines.append(f"2. **Build Models:** Implement {top_models} in Python + PyTorch/scikit-learn using {top_methods} with identical hyperparameters.")
                lines.append("3. **Run Multi-Seed Experiments:** Train and evaluate each model across 3–5 random seeds (e.g., seeds 42, 43, 44).")
                lines.append("4. **Compare & Document:** Report mean ± std accuracy/metrics, paired significance test, and note any dataset or compute limits.")
            else:
                if result.candidate_gaps:
                    lines.append("**What you can build from the Gap Matrix:**")
                    for g in result.candidate_gaps[:4]:
                        lines.append(f"- **{g['gap_label']}** — not found in the {len(result.papers)} loaded papers.")
                elif result.contradictions:
                    lines.append("**What you can build from the paper comparisons:**")
                    for pair_title, cj in result.contradictions[:3]:
                        lines.append(f"- **Controlled replication of {pair_title}** ({cj.classification}) — test both models on one shared dataset and split.")
                else:
                    lines.append(f"**What you can build from the {len(result.papers)} loaded papers:**")
                    if all_archs or all_datasets:
                        lines.append(
                            f"- Compare **{', '.join(all_archs[:4]) or 'the reported models'}** on "
                            f"**{', '.join(all_datasets[:3]) or 'a shared benchmark dataset'}** under one unified train/test split."
                        )
                if result.experiment_plans:
                    _, plan = result.experiment_plans[0]
                    lines.append(
                        f"**What to use to build it:** Dataset: {plan.dataset_and_split} · Baselines: {', '.join(plan.baselines)} · "
                        f"Metrics: {', '.join(plan.metrics)} · Tools: Python, PyTorch/scikit-learn, and 3+ random seeds."
                    )
                else:
                    lines.append(
                        f"**What to use to build it:** Python + PyTorch or scikit-learn, "
                        f"models ({', '.join(all_archs[:4]) or 'ResNet / ViT / baseline'}), "
                        f"datasets ({', '.join(all_datasets[:3]) or 'benchmark from the papers'}), "
                        f"methods ({', '.join(all_methods[:3]) or 'standard ERM / augmentation'}), and 3–5 random seeds."
                    )
        elif wants_simple and wants_build:
            if result.candidate_gaps:
                top_gap = result.candidate_gaps[0]
                lines.append(
                    f"**In simple terms:** Across the papers we read, researchers used **{top_gap['architecture']}** on some problems "
                    f"and tested on **{top_gap['dataset']}** in other papers, but they did not report testing **{top_gap['architecture']}** on **{top_gap['dataset']}** together."
                )
            if result.experiment_plans:
                _, plan = result.experiment_plans[0]
                lines.append(
                    f"**How you can build & test it:** Take **{plan.dataset_and_split}** and compare **{plan.independent_variable}** "
                    f"against **{', '.join(plan.baselines)}** using **{', '.join(plan.metrics)}**. "
                    f"Run each setup 3 times with different random seeds so the comparison is fair."
                )
        elif wants_simple:
            if result.candidate_gaps:
                lines.append("**Here is what the gaps mean in plain English:**")
                for g in result.candidate_gaps[:4]:
                    lines.append(
                        f"- **{g['gap_label']}**: The papers mention using *{g['architecture']}* and testing on *{g['dataset']}* separately, "
                        f"but this specific combination was not reported in the papers we read."
                    )
                lines.append("Note: A 'candidate gap' just means it wasn't in these papers—always double-check outside literature before claiming it is brand new.")
        elif wants_build:
            if result.candidate_gaps:
                top_gaps = [g["gap_label"] for g in result.candidate_gaps[:3]]
                lines.append(f"**Candidate gaps to build on:** {'; '.join(top_gaps)}.")
            if result.experiment_plans:
                _, plan = result.experiment_plans[0]
                steps_preview = " ".join(plan.procedure[:5])
                lines.append(
                    f"**Recommended build ({plan.research_question}):**\n"
                    f"- Dataset: {plan.dataset_and_split}\n"
                    f"- Baselines: {', '.join(plan.baselines)}\n"
                    f"- Metrics: {', '.join(plan.metrics)}\n"
                    f"- Steps: {steps_preview}"
                )
        else:
            if result.gap_assessments:
                lines.append(f"**Found {len(result.candidate_gaps)} candidate gaps in this corpus:**")
                for gap_label, assessment in result.gap_assessments[:4]:
                    lines.append(f"- **{gap_label}** (rating: *{assessment.rating}*): {assessment.reason}")
            elif result.candidate_gaps:
                lines.append(f"Candidate gaps found: {', '.join(g['gap_label'] for g in result.candidate_gaps[:4])}.")

        passages_str, _ = retrieve_passages(standalone_q, result.all_chunks, top_k=3)
        return "\n\n".join(lines), passages_str, f"Grounded in Analysis Results · Intent: {intent}"

    # Factual Q&A across sub-queries via Prompt 08
    combined_query = " ".join(rw.sub_queries)
    passages_str, _ = retrieve_passages(combined_query, result.all_chunks, top_k=5)
    ans_res = runner.run_evidence_answer(
        question=standalone_q,
        passages=passages_str,
        llm_fn=llm_fn,
    )
    ans_text = ans_res.data.text if hasattr(ans_res.data, "text") else str(ans_res.data)
    if rw.needs_external_search:
        ans_text += "\n\n*Note: External search (arXiv/OpenAlex) may be needed if this topic goes beyond the loaded papers.*"

    if ans_res.prompt_used:
        mode_note = f"Prompt v{ans_res.prompt_version} · Intent: {intent} · {'Rule-Based Fallback' if ans_res.used_fallback else 'Validated LLM'}"
    else:
        mode_note = f"Rule-Based (No LLM) · Intent: {intent}"
    return ans_text, passages_str, mode_note
