"""
HypothesisForge - Topic & Scope Advisor
Conducts a short scope interview (team size, timeline, compute, goal) and recommends
3-5 project options labelled 'mini' or 'major', grounded strictly in the Gap Matrix.
"""

from dataclasses import dataclass
from typing import List, Literal

from .pipeline import AnalysisPipelineResult


@dataclass
class ScopeInterview:
    team_size: int = 2
    timeline_weeks: int = 6
    compute_budget: Literal["CPU / Colab Free", "Single GPU", "Multi-GPU"] = "CPU / Colab Free"
    goal: Literal["Mini Project", "Major Project", "Any"] = "Mini Project"


@dataclass
class ScopeRecommendation:
    gap_label: str
    scope_label: Literal["mini", "major"]
    motivating_papers: List[str]
    required_compute: str
    risks: List[str]
    fit_reason: str


MAJOR_DATASETS = {"domainnet", "imagenet", "coco", "mmlu", "wikitext"}
HEAVY_MODELS = {"vit", "vision transformer", "gpt", "llama", "bert", "resnet-101"}


def _classify_gap_scope(arch: str, dataset: str) -> Literal["mini", "major"]:
    a_low = arch.lower()
    d_low = dataset.lower()
    if any(d in d_low for d in MAJOR_DATASETS) or any(m in a_low for m in HEAVY_MODELS):
        return "major"
    return "mini"


def recommend_scoped_topics(
    interview: ScopeInterview,
    result: AnalysisPipelineResult,
    max_options: int = 5,
) -> List[ScopeRecommendation]:
    """
    Recommends 3-5 scoped project options strictly from result.candidate_gaps (never model guesses).
    """
    recommendations: List[ScopeRecommendation] = []

    for gap in result.candidate_gaps:
        gap_label = gap["gap_label"]
        arch = gap["architecture"]
        dataset = gap["dataset"]
        scope = _classify_gap_scope(arch, dataset)

        # Find motivating papers from literature_records that mention arch or dataset
        motivating: List[str] = []
        for pid, rec in result.literature_records.items():
            archs_low = [a.lower() for a in rec.architectures]
            ds_low = [d.lower() for d in rec.datasets]
            if any(arch.lower() in a or a in arch.lower() for a in archs_low) or dataset.lower() in ds_low:
                motivating.append(pid)

        if not motivating:
            motivating = [p.paper_id for p in result.papers[:1]]

        if scope == "mini":
            req_compute = "Low (Colab Free / single entry GPU, ~2-6 hours for 3-5 seeds)"
            risks = [
                "Small benchmark size may limit generalization claims beyond this dataset.",
                "External literature outside this corpus may already cover this pairing.",
            ]
        else:
            req_compute = "Moderate-to-High (Single 16GB+ GPU or Multi-GPU; ~15-40+ GPU hours)"
            risks = [
                "Large dataset/backbone training time may exceed a short mini-project timeline.",
                "Pretraining overlap with test domains can confound out-of-distribution gains.",
                "External literature outside this corpus may already cover this pairing.",
            ]

        # Score fit against user's interview answers
        score = 0
        if interview.goal == "Mini Project" and scope == "mini":
            score += 3
        elif interview.goal == "Major Project" and scope == "major":
            score += 3
        elif interview.goal == "Any":
            score += 1

        if interview.compute_budget == "CPU / Colab Free" and scope == "mini":
            score += 2
        elif interview.compute_budget in {"Single GPU", "Multi-GPU"} and scope == "major":
            score += 2

        if interview.timeline_weeks <= 6 and scope == "mini":
            score += 2
        elif interview.timeline_weeks >= 8 and scope == "major":
            score += 2

        fit_reason = (
            f"Grounded in gap matrix ({gap_label}); labelled '{scope}' based on {arch} and {dataset} scale. "
            f"Matches a {interview.timeline_weeks}-week timeline with {interview.compute_budget}."
            if score >= 3
            else f"Grounded in gap matrix ({gap_label}); labelled '{scope}'. Requires careful scoping for {interview.compute_budget}."
        )

        recommendations.append(
            ScopeRecommendation(
                gap_label=gap_label,
                scope_label=scope,
                motivating_papers=motivating,
                required_compute=req_compute,
                risks=risks,
                fit_reason=fit_reason,
            )
        )

    # Sort best-fitting options first
    def _sort_key(rec: ScopeRecommendation) -> int:
        pts = 0
        if interview.goal == "Mini Project" and rec.scope_label == "mini":
            pts += 3
        if interview.goal == "Major Project" and rec.scope_label == "major":
            pts += 3
        if interview.compute_budget == "CPU / Colab Free" and rec.scope_label == "mini":
            pts += 2
        return -pts

    recommendations.sort(key=_sort_key)
    return recommendations[:max_options]
