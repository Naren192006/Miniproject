"""
HypothesisForge - Rule-based Fallback Extractors and Evaluators
Used when LLM replies are invalid after retry, or when no LLM is configured.
Supports all Prompts 01 - 08 with strict metric-aware and evidence-grounded rules.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from .models import (
    LiteratureExtractionOutput,
    ReportedResultItem,
    GapRelevanceOutput,
    ContradictionJudgementOutput,
    HypothesisGenerationOutput,
    ScientificCriticOutput,
    ExperimentDesignOutput,
    ReportSummaryOutput,
    EvidenceAnswerOutput,
    QueryRewriteOutput,
    TopicInterviewOutput,
    TopicOptionItem,
    TopicRecommendationOutput,
    ConflictExplainerOutput,
    ClosestPaperItem,
    PriorWorkEvidenceOutput,
    SpokenExplanationOutput,
    GroundedClaimItem,
    GroundedQAOutput,
    count_words,
)


KNOWN_TASKS = [
    "Domain Generalization", "Domain Adaptation", "Out-of-Distribution Generalization",
    "Image Classification", "Object Detection", "Inventory Management",
    "Supply Chain Optimization", "Time Series Forecasting", "Reinforcement Learning",
    "Question Answering", "Text Generation", "Regression", "Classification",
]
KNOWN_ARCHITECTURES = [
    "ResNet-50", "ResNet-18", "ResNet-101", "Vision Transformer", "ViT-B/16", "ViT",
    "DenseNet-121", "MobileNetV2", "CNN", "Transformer", "BERT", "GPT", "Llama",
    "LSTM", "GRU", "MLP", "GNN", "GCN", "XGBoost", "Random Forest",
    "Deep Reinforcement Learning", "Reinforcement Learning", "Q-Learning", "DQN", "PPO",
    "Monte Carlo", "Regress-Later Monte Carlo", "Dynamic Programming",
    "Stochastic Programming", "Linear Programming", "POMDP", "MDP",
    "Base-Stock Policy", "(s, S) Policy", "Newsvendor", "Neural Network",
]
KNOWN_DATASETS = [
    "PACS", "OfficeHome", "Office-Home", "VLCS", "DomainNet", "TerraIncognita", "Camelyon17",
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "MMLU", "GSM8K", "GLUE", "SQuAD", "COCO",
    "M5", "Walmart", "Simulation", "Numerical Experiment", "Numerical Study", "Case Study",
    "Synthetic Data", "Retail Data", "Real-World Data",
]
KNOWN_STRATEGIES = [
    "ERM", "Mixup", "CORAL", "DANN", "SWAD", "Fish", "GroupDRO", "ARM", "MLDG",
    "Domain-adversarial training", "LoRA", "QLoRA", "Fine-tuning", "Zero-shot",
    "Value Iteration", "Policy Gradient", "Approximate Dynamic Programming",
    "Stochastic Optimization", "Heuristic", "Myopic Policy", "Gradient Descent",
]
KNOWN_SHIFTS = [
    "Leave-one-domain-out", "Covariate shift", "Concept shift", "Subpopulation shift",
    "Demand uncertainty", "Partial observability", "Non-stationary", "Distribution shift",
]
KNOWN_METRICS = [
    "Worst-domain accuracy", "Top-1 Accuracy", "Average accuracy", "Accuracy",
    "F1 Score", "AUC", "RMSE", "MAE", "MSE", "MAPE", "Perplexity",
    "Expected Cost", "Total Cost", "Holding Cost", "Service Level", "Fill Rate",
    "Profit", "Reward", "Regret", "Optimality Gap", "Runtime",
]

RESULT_VERBS = [
    "achieves", "achieve", "reaches", "reach", "obtains", "obtain",
    "attains", "attain", "reports", "report", "scores", "yields",
    "reduces", "reduce", "improves", "improve", "outperforms", "outperform",
    "decreases", "increases", "accuracy of", "accuracy is", "mse of",
    "rmse of", "cost by", "gap of", "f1 of", "auc of",
]
NOISE_TERMS = [
    "significance", "at the 5% level", "p <", "p=", "train/val", "trainset",
    "testset", "of the images", "±", "error bar", "margin", "split the data",
    "into training", "validation (",
]


def canonicalize_architecture(name: str) -> str:
    """Merges equivalent architecture labels for matrix and knowledge graph consistency."""
    low = name.lower().strip()
    if low in {"vit", "vit-b/16", "vit-b", "vision transformer"}:
        return "Vision Transformer (ViT)"
    return name


def is_cnn_architecture(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ["cnn", "resnet", "densenet", "mobilenet"])


class RuleBasedLiteratureExtractor:
    """Metric-aware deterministic extractor fallback for Prompt 01 (Stage 'literature')."""

    @classmethod
    def extract(cls, title: str, year: Any, paper_text: str) -> LiteratureExtractionOutput:
        found_tasks = [t for t in KNOWN_TASKS if re.search(r"\b" + re.escape(t) + r"\b", paper_text, re.I)]
        found_archs = [a for a in KNOWN_ARCHITECTURES if re.search(r"\b" + re.escape(a) + r"\b", paper_text, re.I)]
        found_datasets = [d for d in KNOWN_DATASETS if re.search(r"\b" + re.escape(d) + r"\b", paper_text, re.I)]
        found_strats = [s for s in KNOWN_STRATEGIES if re.search(r"\b" + re.escape(s) + r"\b", paper_text, re.I)]
        found_shifts = [sh for sh in KNOWN_SHIFTS if re.search(r"\b" + re.escape(sh) + r"\b", paper_text, re.I)]
        found_metrics = [m for m in KNOWN_METRICS if re.search(r"\b" + re.escape(m) + r"\b", paper_text, re.I)]

        # Split by [Page n]: blocks to keep exact page attribution
        page_blocks = re.split(r"\[Page\s+(\d+)\]:?", paper_text)
        pages_list: List[Tuple[int, str]] = []
        if len(page_blocks) > 1:
            for i in range(1, len(page_blocks), 2):
                pg = int(page_blocks[i])
                txt = page_blocks[i + 1] if i + 1 < len(page_blocks) else ""
                pages_list.append((pg, txt))
        else:
            pages_list = [(1, paper_text)]

        reported_results: List[ReportedResultItem] = []

        for page_num, block_text in pages_list:
            sentences = re.split(r"(?<=[.!?])\s+", block_text)
            for sent in sentences:
                sent_clean = re.sub(r"\s+", " ", sent).strip()
                if not sent_clean or len(sent_clean) < 15:
                    continue
                sent_lower = sent_clean.lower()

                # Reject significance thresholds, data split descriptions, and error bars
                if any(noise in sent_lower for noise in NOISE_TERMS):
                    continue

                # Require a result verb AND an explicit metric name in the same sentence
                if not any(v in sent_lower for v in RESULT_VERBS):
                    continue
                metric_match = next(
                    (m for m in KNOWN_METRICS if re.search(r"\b" + re.escape(m) + r"\b", sent_clean, re.I)),
                    None,
                )
                if not metric_match:
                    continue

                # Prefer numeric value directly following a result verb or metric name in the same sentence
                metric_esc = re.escape(metric_match)
                num_match = re.search(
                    rf"(?:achieves|reaches|obtains|attains|reports|scores|yields|reduces|improves|{metric_esc})\s+(?:of\s+|by\s+|is\s+)?(\d{{1,4}}(?:\.\d{{1,3}})?)\s*%?",
                    sent_clean,
                    re.I,
                )
                if not num_match:
                    num_match = re.search(r"\b(\d{1,4}(?:\.\d{1,3})?)\s*%", sent_clean)
                if not num_match:
                    continue
                val = float(num_match.group(1))
                # Sanity check: ignore obvious non-score numbers like 0.0
                if val <= 0.0:
                    continue

                # Extract dataset, architecture, and strategy ONLY if mentioned in this sentence
                ds_in_sent = next(
                    (d for d in KNOWN_DATASETS if re.search(r"\b" + re.escape(d) + r"\b", sent_clean, re.I)),
                    "not stated",
                )
                arch_in_sent = next(
                    (a for a in KNOWN_ARCHITECTURES if re.search(r"\b" + re.escape(a) + r"\b", sent_clean, re.I)),
                    "not stated",
                )
                strat_in_sent = next(
                    (s for s in KNOWN_STRATEGIES if re.search(r"\b" + re.escape(s) + r"\b", sent_clean, re.I)),
                    "not stated",
                )
                setting_parts = [p for p in [arch_in_sent, strat_in_sent] if p != "not stated"]
                setting_str = ", ".join(setting_parts) if setting_parts else "not stated"

                method_type = "baseline" if "baseline" in sent_lower else "proposed"

                words = sent_clean.split()
                quote = " ".join(words[:25]) if len(words) > 25 else sent_clean

                reported_results.append(
                    ReportedResultItem(
                        value=val,
                        metric=metric_match,
                        dataset=ds_in_sent,
                        setting=setting_str,
                        method=method_type,
                        page=page_num,
                        quote=quote,
                    )
                )
                if len(reported_results) >= 8:
                    break
            if len(reported_results) >= 8:
                break

        return LiteratureExtractionOutput(
            tasks=found_tasks if found_tasks else [],
            architectures=found_archs,
            datasets=found_datasets,
            training_strategies=found_strats,
            shift_settings=found_shifts,
            metrics=found_metrics if found_metrics else [],
            evaluation_protocol="not stated",
            reported_results=reported_results,
            findings=[],
            limitations=[],
            extraction_confidence="low",
        )


class RuleBasedGapEvaluator:
    """Deterministic rule-based fallback for Prompt 02 (Stage 'gaps')."""

    @classmethod
    def evaluate(cls, gap: str, related_records: str) -> GapRelevanceOutput:
        lines = [line.strip() for line in related_records.splitlines() if line.strip()]
        is_thin = len(lines) <= 2 or len(related_records.strip()) < 100

        if is_thin:
            rating = "low"
            reason = (
                f"The evidence in these papers is thin for '{gap}', with too few reported experiments "
                "to judge how meaningful this combination would be."
            )
            absence = "corpus size"
        else:
            rating = "medium"
            reason = (
                f"Across the papers examined, '{gap}' was not tested together even though the individual model "
                "and dataset appear separately. Testing this pairing is a plausible next step to check if results hold."
            )
            absence = "not reported by these papers but plausible"

        questions = [
            "Does prior literature outside this corpus evaluate this specific combination?",
            "Are there standard implementation baselines available for this setting?",
            "Are the computational requirements feasible for this experimental protocol?",
        ]

        return GapRelevanceOutput(
            rating=rating,
            reason=reason,
            absence_explanation=absence,
            questions_to_check=questions,
        )


class RuleBasedContradictionEvaluator:
    """Deterministic rule-based fallback for Prompt 03 (Stage 'contradictions')."""

    @classmethod
    def evaluate(cls, pair_text: str) -> ContradictionJudgementOutput:
        pair_lower = pair_text.lower()

        papers = re.findall(r"-\s*Paper:\s*([^\n]+)", pair_text)
        datasets = re.findall(r"-\s*Dataset:\s*([^\n]+)", pair_text)
        metrics = re.findall(r"-\s*Metric:\s*([^\n]+)", pair_text)
        settings = re.findall(r"-\s*Setting:\s*([^\n]+)", pair_text)
        values = re.findall(r"-\s*Value:\s*([^\n]+)", pair_text)
        p_a = papers[0].strip() if len(papers) > 0 else "Paper A"
        p_b = papers[1].strip() if len(papers) > 1 else "Paper B"
        d_a = datasets[0].strip() if len(datasets) > 0 else "not stated"
        d_b = datasets[1].strip() if len(datasets) > 1 else "not stated"
        m_a = metrics[0].strip() if len(metrics) > 0 else "not stated"
        m_b = metrics[1].strip() if len(metrics) > 1 else "not stated"
        s_a = settings[0].strip() if len(settings) > 0 else "not stated"
        s_b = settings[1].strip() if len(settings) > 1 else "not stated"
        v_a = values[0].strip() if len(values) > 0 else "a reported value"
        v_b = values[1].strip() if len(values) > 1 else "a reported value"

        factors: List[str] = []
        if d_a.lower() != d_b.lower():
            factors.append("dataset")
        if m_a.lower() != m_b.lower():
            factors.append("metric")
        if s_a.lower() != s_b.lower():
            factors.append("hyperparameters")
        if "protocol: not stated" in pair_lower or "split: not stated" in pair_lower:
            factors.extend(["split", "seed"])
        if not factors:
            factors.append("seed")
        factors = list(dict.fromkeys(factors))

        known_found = [d for d in KNOWN_DATASETS if d.lower() in pair_lower]
        if len(set(known_found)) > 1 or (
            d_a != d_b
            and "unknown" not in d_a.lower()
            and "not stated" not in d_a.lower()
            and "unknown" not in d_b.lower()
            and "not stated" not in d_b.lower()
        ):
            conf = 0.88 if ("protocol: not stated" not in pair_lower) else 0.78
            return ContradictionJudgementOutput(
                classification="non-comparable evidence",
                explanation=(
                    f"These two studies are not contradicting each other because they tested on different datasets: "
                    f"{p_a} evaluated on {d_a} ({v_a}), while {p_b} evaluated on {d_b} ({v_b}). "
                    f"Different datasets have different difficulty levels and data distributions, so their scores cannot be compared directly."
                ),
                conditions_to_verify=[
                    f"Check whether {p_a} and {p_b} were ever tested on the same benchmark dataset",
                    "Confirm whether both papers used the same data splits and preprocessing",
                    "Verify if the same model architecture and training budget were used",
                ],
                confidence_score=conf,
                conflict_factors=factors,
            )

        if (
            "protocol: not stated" in pair_lower
            or "split: not stated" in pair_lower
            or "unknown" in d_a.lower()
            or "not stated" in d_a.lower()
            or "unknown" in d_b.lower()
            or "not stated" in d_b.lower()
        ):
            return ContradictionJudgementOutput(
                classification="insufficient information",
                explanation=(
                    f"We cannot tell if {p_a} ({d_a}, {v_a}) and {p_b} ({d_b}, {v_b}) truly disagree yet because key testing details "
                    f"(such as the exact dataset, train/test split, or evaluation protocol) are missing or unstated. "
                    f"Until those testing conditions are confirmed in the original papers, this is marked as needing more information rather than a true conflict."
                ),
                conditions_to_verify=[
                    "Check which exact dataset and train/test split each paper used",
                    "Verify how many random runs/seeds were averaged",
                    "Confirm that both papers measure the exact same metric definition",
                ],
                confidence_score=0.45,
                conflict_factors=factors,
            )

        return ContradictionJudgementOutput(
            classification="apparent agreement",
            explanation=(
                f"{p_a} ({v_a}) and {p_b} ({v_b}) appear consistent on {d_a} under comparable settings, "
                f"with no direct conflict detected in the available records."
            ),
            conditions_to_verify=[
                "Confirm seed count and variance reporting",
                "Verify optimizer tuning budget",
            ],
            confidence_score=0.72,
            conflict_factors=factors,
        )


class RuleBasedHypothesisGenerator:
    """Deterministic rule-based fallback for Prompt 04 (Stage 'hypotheses')."""

    @classmethod
    def generate(cls, gap: str) -> HypothesisGenerationOutput:
        gap_header = next((line.strip() for line in gap.splitlines() if line.strip()), "")
        # Extract "<Arch> on <Dataset>" directly from gap header if present
        header_clean = re.sub(r"^(?:Candidate\s*)?Gap:\s*", "", gap_header, flags=re.I).strip()
        if " on " in header_clean:
            parts = header_clean.split(" on ", 1)
            found_arch = parts[0].strip()
            found_dataset = parts[1].strip().split()[0]
        else:
            found_arch = next((a for a in KNOWN_ARCHITECTURES if a.lower() in gap_header.lower()), "")
            found_dataset = next((d for d in KNOWN_DATASETS if d.lower() in gap_header.lower()), "")

        found_metric = next((m for m in KNOWN_METRICS if m.lower() in gap.lower()), "Performance")

        # Only cite paper IDs from lines that actually mention the gap's architecture or dataset
        relevant_ids: List[str] = []
        for line in gap.splitlines():
            line_pids = re.findall(r"\b(P-[A-Z0-9]{6}|P\d+|Paper\s*\d+)\b", line, re.IGNORECASE)
            if not line_pids:
                continue
            line_low = line.lower()
            if (found_arch and found_arch.lower() in line_low) or (found_dataset and found_dataset.lower() in line_low):
                relevant_ids.extend(line_pids)

        if not relevant_ids or not found_arch or not found_dataset:
            return HypothesisGenerationOutput(
                decision="insufficient_evidence",
                reason=f"Insufficient grounded evidence: no paper in the corpus links both {found_arch or 'an architecture'} and a comparable baseline on {found_dataset or 'this dataset'}."
            )

        relevant_ids = list(dict.fromkeys(relevant_ids))

        # Avoid comparing CNN against ResNet-50 (since ResNet-50 IS a CNN)
        if is_cnn_architecture(found_arch):
            baseline_arch = "Vision Transformer (ViT) ERM"
        else:
            baseline_arch = "ResNet-50 ERM"

        # Rule: Use "unknown" unless a quoted passage supports a direction
        statement = (
            f"If {found_arch} is evaluated on {found_dataset}, "
            f"then {found_metric} may differ, measured by {found_metric}, relative to {baseline_arch} on {found_dataset}."
        )

        return HypothesisGenerationOutput(
            decision="hypothesis",
            statement=statement,
            expected_direction="unknown",
            baseline=f"{baseline_arch} on {found_dataset}",
            metric=found_metric,
            assumptions=[
                "Consistent preprocessing and pretraining weights are used across compared backbones",
                "Evaluation follows the dataset's standard out-of-distribution split",
                "Hyperparameter tuning budgets are matched across models",
            ],
            cited_paper_ids=relevant_ids,
            reason=(
                f"Cited paper(s) {', '.join(relevant_ids)} report experiments involving {found_arch} or {found_dataset}, "
                f"supporting a testable comparison against {baseline_arch}."
            ),
        )


class RuleBasedScientificCritic:
    """Deterministic rule-based fallback for Prompt 05 (Stage 'critic')."""

    @classmethod
    def evaluate(
        cls,
        hypothesis: str,
        checks: str,
        prior_status: str,
        overlap: str,
    ) -> ScientificCriticOutput:
        checks_lower = checks.lower()
        prior_lower = prior_status.lower().strip()
        overlap_lower = overlap.lower().strip()

        if "cannot be traced" in checks_lower or "unsupported" in checks_lower or "failed" in checks_lower:
            return ScientificCriticOutput(
                verdict="reject",
                rationale="The automated checks indicate the proposed claim cannot be traced to verifiable evidence in the corpus papers. Unsupported claims cannot proceed to experiment design without grounding.",
                missing_evidence=[
                    "Direct quotes establishing baseline metric on target dataset",
                    "Evidence linking the proposed intervention to the chosen architecture",
                ],
                alternative_explanations=[
                    "Reporting bias favoring positive outcomes in benchmark literature",
                    "Uncontrolled difference in data preprocessing pipeline",
                ],
            )

        has_overlap = bool(overlap_lower) and not any(neg in overlap_lower for neg in ["none", "empty", "no overlap", "[]"])
        if prior_lower in ["not run", "not_run", "skipped", "pending"] or has_overlap:
            return ScientificCriticOutput(
                verdict="revise",
                rationale=(
                    "The hypothesis is plausible and testable, but external prior-work search was not run (or indicated potential overlap). "
                    "Per review rules, a claim cannot be accepted until prior literature outside this corpus is checked."
                ),
                missing_evidence=[
                    "External literature search confirming whether this pairing has already been benchmarked",
                    "Verified baseline protocol details from the original papers",
                ],
                alternative_explanations=[
                    "Pretraining dataset overlap with target benchmark test domains",
                    "Differences in parameter count or tuning budget between architectures",
                ],
            )

        return ScientificCriticOutput(
            verdict="accept",
            rationale="The hypothesis is specific, measurable, and falsifiable with clear baseline comparisons. Automated checks confirm traceability and the prior-work search found no covering literature.",
            missing_evidence=[],
            alternative_explanations=[
                "Domain-specific inductive bias differences between convolution and attention",
                "Implicit regularization benefits from pretraining scale",
            ],
        )


class RuleBasedExperimentDesigner:
    """Deterministic rule-based fallback for Prompt 06 (Stage 'experiments')."""

    @classmethod
    def design(
        cls,
        hypothesis: str,
        draft_plan: str,
        grounded_options: str,
    ) -> ExperimentDesignOutput:
        grounded_lower = grounded_options.lower()
        hyp_lower = hypothesis.lower()

        if_match = re.search(r"If\s+(.+?)\s+is\s+evaluated\s+on\s+([^,]+)", hypothesis, re.I)
        if if_match:
            arch_name = if_match.group(1).strip()
            dataset_name = if_match.group(2).strip()
        else:
            dataset_name = next((d for d in KNOWN_DATASETS if d.lower() in hyp_lower), None)
            if not dataset_name:
                dataset_name = "PACS" if "pacs" in grounded_lower else "OfficeHome"
            arch_name = next((a for a in KNOWN_ARCHITECTURES if a.lower() in hyp_lower), "Proposed Architecture")

        # Separate independent variable from baseline cleanly
        if is_cnn_architecture(arch_name):
            baseline_name = "Vision Transformer (ViT) ERM"
        else:
            baseline_name = "ResNet-50 ERM" if "resnet-50" in grounded_lower or "erm" in grounded_lower else "Standard ERM Baseline (verify)"

        if dataset_name.lower() == "camelyon17":
            dataset_split = "Camelyon17 using hospital-based out-of-distribution split (verify exact WILDS split against source)."
        else:
            dataset_split = f"{dataset_name} using the standard out-of-distribution split reported in the source papers (verify split)."

        return ExperimentDesignOutput(
            research_question=f"Does {arch_name} differ in out-of-domain Accuracy on {dataset_name} relative to {baseline_name}?",
            independent_variable=f"Model architecture backbone ({arch_name} vs {baseline_name})",
            dependent_variable="Out-of-domain Accuracy (%)",
            dataset_and_split=dataset_split,
            baselines=[baseline_name, "Mixup (verify)"],
            controls=[
                "Identical training optimizer (AdamW, lr=1e-4)",
                "Equal training epochs and batch sizes across all conditions",
                "Consistent data augmentation and normalization pipeline",
            ],
            metrics=["Average Accuracy", "Worst-domain Accuracy"],
            procedure=[
                f"1. Download and preprocess {dataset_name} benchmark splits.",
                f"2. Initialize the baseline model ({baseline_name}) with pretrained weights.",
                "3. Train the baseline across all target splits using at least three random seeds.",
                f"4. Train {arch_name} under identical hyperparameter conditions across the same random seeds.",
                "5. Evaluate out-of-domain test accuracy and report mean, standard deviation, and paired comparison.",
            ],
            seeds_and_runs="Three random seeds (seeds 42, 43, 44) repeated across all target splits (increase to 5+ seeds if compute permits).",
            statistics="Two-tailed paired t-test across random seeds (note: 3 seeds has very low statistical power; treat p-values as exploratory).",
            compute_estimate="Unknown (compute estimate depends on hardware, image resolution, and dataset size).",
            threats_to_validity=[
                "Data leakage: Potential overlap between pretraining datasets and target domains.",
                "Confounder: Disparity in parameter count and pretraining scale between backbones.",
                "Tuning budget: Unequal hyperparameter exploration between compared models.",
            ],
            expected_observations="Higher target-domain accuracy for the tested backbone would indicate stronger cross-domain generalization under matched budgets, whereas parity or lower accuracy would suggest the baseline remains preferable.",
            limitations=[
                f"Evaluation restricted to {dataset_name} distribution shifts.",
                "Using only three random seeds yields low statistical power.",
            ],
        )


class RuleBasedReportSummarizer:
    """Deterministic rule-based fallback for Prompt 07 (Stage 'report')."""

    @classmethod
    def summarize(cls, facts: str) -> ReportSummaryOutput:
        try:
            facts_dict = json.loads(facts)
        except Exception:
            facts_dict = {}

        paper_count = facts_dict.get("paper_count", "several")
        topic = facts_dict.get("topic", "").strip() or "the selected research topic"
        datasets = facts_dict.get("datasets", [])
        protocols_stated = facts_dict.get("protocols_stated_count", 0)
        gaps_count = facts_dict.get("candidate_gaps_count", 0)
        declined_count = facts_dict.get("hypotheses_declined", 0)
        unrun = facts_dict.get("unrun_analyses", [])

        ds_text = ", ".join(datasets[:4]) if datasets else "the reported benchmarks"
        proto_note = (
            "evaluation protocols were extracted where stated"
            if protocols_stated
            else "evaluation protocols were mostly not stated in the extracted records"
        )
        p1 = (
            f"The examined corpus covers {topic} across {paper_count} papers "
            f"referencing {ds_text}. Automated extraction recorded available models and datasets, though {proto_note}."
        )

        decline_note = (
            f"{declined_count} candidate hypothesis was declined for insufficient evidence."
            if declined_count
            else "no candidate hypotheses were declined, though all remain tentative."
        )
        p2 = (
            f"Analysis surfaced {gaps_count} candidate gaps and checked cross-paper candidate contradictions, "
            f"marking pairs with missing splits as insufficient information. Additionally, {decline_note}"
        )

        unrun_text = ", ".join(unrun) if unrun else "external prior-work search"
        p3 = (
            f"Before running experiments, researchers should verify original evaluation splits and baseline details in the source PDFs. "
            f"Note that {unrun_text} could not run during this pass and must be checked manually."
        )

        full_text = f"{p1}\n\n{p2}\n\n{p3}"
        return ReportSummaryOutput(
            text=full_text,
            paragraphs=[p1, p2, p3],
            word_count=count_words(full_text),
        )


class RuleBasedEvidenceAnswerer:
    """Deterministic rule-based fallback for Prompt 08 ('Ask the papers')."""

    @classmethod
    def answer(cls, question: str, passages: str) -> EvidenceAnswerOutput:
        if not passages.strip() or "no passages" in passages.lower():
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        passage_blocks = re.findall(r"\[(?:Passage\s*)?(\d+)\]\s*([^\[]+)", passages)
        if not passage_blocks:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        stop_words = {
            "what", "which", "does", "that", "this", "with", "from", "have", "about",
            "they", "their", "were", "when", "where", "how", "build", "make", "using",
            "want", "need", "give", "tell", "show", "explain", "paper", "papers", "any",
            "did", "get", "was", "are", "the", "and", "for",
        }
        synonyms = {
            "accuracy": {"accuracy", "acc", "auc", "f1", "score", "performance", "precision", "recall"},
            "dataset": {"dataset", "datasets", "data", "benchmark", "corpus"},
            "method": {"method", "methods", "approach", "technique", "preprocessing", "pre-processing", "algorithm"},
            "model": {"model", "models", "architecture", "network", "backbone", "cnn", "transformer"},
            "result": {"result", "results", "outcome", "achieved", "reached", "reported"},
        }
        question_words = {
            w for w in re.findall(r"[a-zA-Z0-9\-]{3,}", question.lower())
            if w not in stop_words
        }
        if not question_words:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        # Expand with stem + domain synonyms for better recall
        expanded_groups: List[set] = []
        for w in question_words:
            grp = {w}
            w_norm = re.sub(r"[-_]", "", w)
            if len(w_norm) >= 5:
                grp.add(w_norm[: max(4, len(w_norm) - 2)])
            for syn_key, syn_set in synonyms.items():
                if w in syn_set or w == syn_key:
                    grp |= syn_set
            expanded_groups.append(grp)

        # Score individual sentences inside each passage
        scored_sentences: List[Tuple[float, int, str]] = []
        for p_num_str, p_text in passage_blocks:
            p_num = int(p_num_str)
            cleaned = re.sub(r"^\([^)]+\):\s*", "", p_text).strip()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue
                s_low = sent_clean.lower()
                s_norm = re.sub(r"[-_]", "", s_low)
                hits = 0.0
                for grp in expanded_groups:
                    if any(term in s_low or term in s_norm for term in grp):
                        hits += 1.0
                if hits > 0:
                    scored_sentences.append((hits, p_num, sent_clean))

        if not scored_sentences:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        min_hits = 1.0 if len(question_words) <= 3 else 2.0
        top_sentences = [item for item in scored_sentences if item[0] >= min_hits][:3]
        if not top_sentences:
            top_sentences = scored_sentences[:2]

        sentences_out: List[str] = []
        cited_nums: List[int] = []
        seen_nums = set()
        for _, p_num, sent_clean in top_sentences:
            sent_no_dot = sent_clean.rstrip(".!?")
            sentences_out.append(f"{sent_no_dot} [{p_num}].")
            if p_num not in seen_nums:
                cited_nums.append(p_num)
                seen_nums.add(p_num)

        ans_text = " ".join(sentences_out)
        return EvidenceAnswerOutput(
            text=ans_text,
            paragraphs=[ans_text],
            citations=cited_nums,
            is_declined=False,
        )


class RuleBasedQueryRewriter:
    """Deterministic fallback for Prompt 09 (Query Rewriter & Intent Router)."""

    @classmethod
    def rewrite(cls, history: str, question: str) -> QueryRewriteOutput:
        q_clean = question.strip()
        q_low = q_clean.lower()

        # Resolve simple pronouns ("it", "that dataset", "that model") from chat history
        standalone = q_clean
        if history.strip() and re.search(r"\b(it|that dataset|that model|that paper|those)\b", q_low):
            hist_entities = (
                re.findall(r"\b(" + "|".join(re.escape(x) for x in KNOWN_DATASETS + KNOWN_ARCHITECTURES) + r")\b", history, re.I)
            )
            if hist_entities:
                last_entity = hist_entities[-1]
                standalone = re.sub(r"\b(that dataset|that model|it)\b", last_entity, standalone, count=1, flags=re.I)

        # Typo-tolerant intent matching (e.g., "summaarize" -> "summarize")
        import difflib
        q_words = re.findall(r"[a-z]+", q_low)
        def _has_fuzzy(targets: List[str], cutoff: float = 0.78) -> bool:
            return any(t in q_low for t in targets) or any(
                difflib.get_close_matches(w, targets, n=1, cutoff=cutoff) for w in q_words if len(w) >= 5
            )

        # Classify intent
        if any(k in q_low for k in ["weather", "capital of", "recipe", "rocket", "pizza", "movie", "football"]):
            intent = "out_of_scope"
        elif _has_fuzzy(["contradict", "contradiction", "disagree", "conflict"]):
            intent = "contradiction"
        elif _has_fuzzy(["scope", "mini", "major", "topic_advice", "which topic", "recommend a topic", "what can i build", "what should i build", "what should i use", "project idea", "roadmap", "step by step", "step-by-step", "how to start", "action plan"]):
            intent = "topic_advice"
        elif _has_fuzzy(["gap", "gaps", "untested", "missing combination", "build"]):
            intent = "gap"
        elif _has_fuzzy(["compare", "versus", "difference between"]) or " vs " in q_low:
            intent = "compare"
        elif _has_fuzzy(["summary", "summarize", "summarise", "overview"]) or any(
            p in q_low for p in ["what are the papers about", "about these papers", "explain the papers"]
        ):
            intent = "summary"
        else:
            intent = "factual_qa"

        # Split multi-part questions into up to 3 sub-queries
        raw_parts = [p.strip() for p in re.split(r"[?;]|\band\b", standalone) if len(p.strip()) >= 5]
        sub_queries = raw_parts[:3] if raw_parts else [standalone]

        needs_external = intent == "out_of_scope" or any(
            k in q_low for k in ["outside", "external", "search arxiv", "search openalex"]
        )

        return QueryRewriteOutput(
            standalone_question=standalone,
            intent=intent,  # type: ignore
            sub_queries=sub_queries,
            needs_external_search=needs_external,
        )


class RuleBasedTopicInterviewer:
    """Deterministic fallback for Prompt 10 (Project-Topic Interviewer)."""

    REQUIRED_FIELDS: List[Tuple[str, str]] = [
        ("field_of_interest", "What research area or topic are you most interested in exploring?"),
        ("project_level", "Are you planning a mini project, a major project, or a thesis?"),
        ("compute_access", "What compute hardware do you have access to (none, laptop CPU, or GPU)?"),
        ("available_weeks", "How many weeks do you have available to complete this project?"),
        ("current_skills", "What programming or machine learning skills do you already have?"),
        ("preferred_outcome", "What outcome do you want most (e.g., working prototype, benchmark study, or paper)?"),
    ]

    @classmethod
    def interview(cls, answers: Any) -> TopicInterviewOutput:
        parsed: Dict[str, Any] = {}
        if isinstance(answers, dict):
            parsed = dict(answers)
        elif isinstance(answers, str) and answers.strip():
            try:
                loaded = json.loads(answers)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                for line in answers.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed[k.strip().lower()] = v.strip()

        normalized = {str(k).strip().lower(): v for k, v in parsed.items()}
        missing: List[str] = []
        next_q = ""
        for field, question in cls.REQUIRED_FIELDS:
            val = normalized.get(field)
            if val is None or str(val).strip() == "":
                missing.append(field)
                if not next_q:
                    next_q = question

        if not missing:
            return TopicInterviewOutput(ready=True, next_question="", missing_fields=[])
        return TopicInterviewOutput(ready=False, next_question=next_q, missing_fields=missing)


class RuleBasedTopicRecommender:
    """Deterministic fallback for Prompt 11 (Topic & Scope Recommender)."""

    MAJOR_DATASETS = {"domainnet", "imagenet", "coco", "mmlu", "wikitext"}
    HEAVY_MODELS = {"vit", "vision transformer", "gpt", "llama", "bert", "resnet-101"}

    @classmethod
    def recommend(
        cls,
        gaps: Any,
        contradictions: Any,
        profile: Any = "",
        scope_criteria: Any = "",
        n_papers: int = 0,
    ) -> TopicRecommendationOutput:
        # Parse gaps list
        gap_items: List[Dict[str, Any]] = []
        if isinstance(gaps, list):
            gap_items = [g for g in gaps if isinstance(g, dict)]
        elif isinstance(gaps, str) and gaps.strip():
            try:
                loaded = json.loads(gaps)
                if isinstance(loaded, list):
                    gap_items = [g for g in loaded if isinstance(g, dict)]
            except Exception:
                for line in gaps.splitlines():
                    line = line.strip("-• ").strip()
                    if " on " in line:
                        arch, ds = line.split(" on ", 1)
                        gap_items.append({"architecture": arch.strip(), "dataset": ds.strip(), "gap_label": line})

        # Parse contradictions list
        con_items: List[Dict[str, Any]] = []
        if isinstance(contradictions, list):
            con_items = [c for c in contradictions if isinstance(c, dict)]
        elif isinstance(contradictions, str) and contradictions.strip():
            try:
                loaded = json.loads(contradictions)
                if isinstance(loaded, list):
                    con_items = [c for c in loaded if isinstance(c, dict)]
            except Exception:
                pass

        criteria_low = json.dumps(scope_criteria).lower() if isinstance(scope_criteria, dict) else str(scope_criteria).lower()
        prefer_mini = "mini" in criteria_low and "major" not in criteria_low
        prefer_major = "major" in criteria_low and "mini" not in criteria_low

        # Parse effort weeks cap if available
        weeks_cap = 8
        prof_str = json.dumps(profile) if isinstance(profile, dict) else str(profile)
        wk_match = re.search(r"available_weeks[\"':\s]+(\d+)", prof_str, re.I)
        if wk_match:
            weeks_cap = max(2, min(24, int(wk_match.group(1))))

        options: List[TopicOptionItem] = []
        n_label = max(1, int(n_papers or 0))

        for g in gap_items:
            arch = str(g.get("architecture", "Proposed Model")).strip()
            ds = str(g.get("dataset", "Benchmark")).strip()
            label = str(g.get("gap_label") or f"{arch} on {ds}")
            ev_ids = [str(x) for x in (g.get("supporting_paper_ids") or g.get("evidence_ids") or []) if str(x).strip()]
            if not ev_ids:
                ev_ids = ["P-CORPUS"]

            is_heavy = any(d in ds.lower() for d in cls.MAJOR_DATASETS) or any(m in arch.lower() for m in cls.HEAVY_MODELS)
            scope: str = "major" if is_heavy else "mini"
            if prefer_mini:
                scope = "mini"
            elif prefer_major:
                scope = "major"

            effort = min(weeks_cap, 4 if scope == "mini" else 10)
            compute = (
                "Laptop CPU or Colab Free (approx. 2-6 hours for 3 seeds)"
                if scope == "mini"
                else "Single 16GB+ GPU or Multi-GPU (approx. 15-35 GPU hours)"
            )
            options.append(
                TopicOptionItem(
                    title=f"Evaluate {label}",
                    scope=scope,  # type: ignore
                    reason=f"The pairing {label} was not found in the {n_label} papers searched and can be tested against reported baselines.",
                    evidence_ids=ev_ids,
                    datasets=[ds],
                    compute_needed=compute,
                    papers_to_read=ev_ids,
                    risks=[
                        f"External literature outside the {n_label} papers searched may cover this pairing.",
                        "Split or hyperparameter mismatch with source papers can confound comparisons.",
                    ],
                    effort_weeks=max(1, effort),
                )
            )
            if len(options) >= 5:
                break

        # Add contradiction-traceable option if room remains
        for c in con_items:
            if len(options) >= 5:
                break
            pa = str(c.get("paper_a") or c.get("paper_a_id") or "").strip()
            pb = str(c.get("paper_b") or c.get("paper_b_id") or "").strip()
            ds = str(c.get("dataset") or "Benchmark").strip()
            ev_ids = [x for x in [pa, pb] if x] or ["P-CORPUS"]
            scope = "mini" if prefer_mini or not any(d in ds.lower() for d in cls.MAJOR_DATASETS) else "major"
            options.append(
                TopicOptionItem(
                    title=f"Controlled replication of {pa or 'Paper A'} vs {pb or 'Paper B'} on {ds}",
                    scope=scope,  # type: ignore
                    reason=f"Resolves conflicting results across {', '.join(ev_ids)} under a unified split; controlled comparison was not found in the {n_label} papers searched.",
                    evidence_ids=ev_ids,
                    datasets=[ds],
                    compute_needed="Laptop CPU or Single GPU (approx. 3-8 hours)",
                    papers_to_read=ev_ids,
                    risks=["Original code or random seeds may differ from published summaries."],
                    effort_weeks=max(1, min(weeks_cap, 4 if scope == "mini" else 8)),
                )
            )

        warnings: List[str] = []
        if len(options) < 3:
            warnings.append(
                f"Evidence is too thin for 3 options: only {len(options)} traceable gap/contradiction candidate(s) found in the {n_label} papers searched."
            )

        return TopicRecommendationOutput(options=options[:5], warnings=warnings)


class RuleBasedConflictExplainer:
    """Deterministic fallback for Prompt 12 (Condition-Aware Conflict Explainer)."""

    @staticmethod
    def _parse_record(rec: Any) -> Dict[str, Any]:
        if isinstance(rec, dict):
            return {str(k).strip().lower(): v for k, v in rec.items()}
        parsed: Dict[str, Any] = {}
        if isinstance(rec, str) and rec.strip():
            try:
                loaded = json.loads(rec)
                if isinstance(loaded, dict):
                    return {str(k).strip().lower(): v for k, v in loaded.items()}
            except Exception:
                for line in rec.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed[k.strip().lower()] = v.strip()
        return parsed

    @staticmethod
    def _clip_quote(q: Any) -> str:
        s = str(q or "").strip()
        if not s:
            return "N/A"
        words = s.split()
        return " ".join(words[:25])

    @staticmethod
    def _safe_page(p: Any) -> int:
        try:
            return max(0, int(p))
        except Exception:
            return 0

    @classmethod
    def explain(cls, record_a: Any, record_b: Any) -> ConflictExplainerOutput:
        a = cls._parse_record(record_a)
        b = cls._parse_record(record_b)

        quote_a = cls._clip_quote(a.get("verbatim_quote") or a.get("quote"))
        quote_b = cls._clip_quote(b.get("verbatim_quote") or b.get("quote"))
        page_a = cls._safe_page(a.get("page"))
        page_b = cls._safe_page(b.get("page"))

        ds_a, ds_b = str(a.get("dataset", "")).strip(), str(b.get("dataset", "")).strip()
        met_a, met_b = str(a.get("metric", "")).strip(), str(b.get("metric", "")).strip()
        spl_a, spl_b = str(a.get("split", "")).strip(), str(b.get("split", "")).strip()

        if not (ds_a and ds_b and met_a and met_b and spl_a and spl_b):
            return ConflictExplainerOutput(
                verdict="insufficient_info",
                differing_factors=[],
                quote_a=quote_a,
                page_a=page_a,
                quote_b=quote_b,
                page_b=page_b,
                confidence="low",
                explanation="One or both records are missing dataset, metric, or split information needed to compare conditions.",
            )

        differing: List[str] = []
        if ds_a.lower() != ds_b.lower():
            differing.append("dataset")
        if met_a.lower() != met_b.lower():
            differing.append("metric")
        if spl_a.lower() != spl_b.lower():
            differing.append("split")

        arch_a, arch_b = str(a.get("architecture", "")).strip(), str(b.get("architecture", "")).strip()
        hyp_a, hyp_b = str(a.get("hyperparameters", "")).strip(), str(b.get("hyperparameters", "")).strip()
        seed_a, seed_b = str(a.get("seed", "")).strip(), str(b.get("seed", "")).strip()

        if arch_a and arch_b and arch_a.lower() != arch_b.lower():
            differing.append("architecture")
        if hyp_a and hyp_b and hyp_a.lower() != hyp_b.lower():
            differing.append("hyperparameters")
        if seed_a and seed_b and seed_a.lower() != seed_b.lower():
            differing.append("seed")

        core_mismatch = any(f in differing for f in ("dataset", "metric", "split"))
        if core_mismatch:
            return ConflictExplainerOutput(
                verdict="condition_mismatch",
                differing_factors=differing,  # type: ignore
                quote_a=quote_a,
                page_a=page_a,
                quote_b=quote_b,
                page_b=page_b,
                confidence="high",
                explanation=f"Core evaluation conditions differ ({', '.join(differing)}), so the reported values are not directly comparable.",
            )

        # Dataset, metric, and split all match
        val_a = str(a.get("value", "")).strip()
        val_b = str(b.get("value", "")).strip()
        if val_a and val_b and val_a.lower() != val_b.lower():
            conf = "medium" if differing else "high"
            return ConflictExplainerOutput(
                verdict="true_conflict",
                differing_factors=differing,  # type: ignore
                quote_a=quote_a,
                page_a=page_a,
                quote_b=quote_b,
                page_b=page_b,
                confidence=conf,  # type: ignore
                explanation=f"Dataset ({ds_a}), metric ({met_a}), and split ({spl_a}) match while values differ ({val_a} vs {val_b}).",
            )

        return ConflictExplainerOutput(
            verdict="insufficient_info",
            differing_factors=differing,  # type: ignore
            quote_a=quote_a,
            page_a=page_a,
            quote_b=quote_b,
            page_b=page_b,
            confidence="low",
            explanation="Conditions match, but differing conclusions or values could not be confirmed from the provided fields.",
        )


class RuleBasedPriorWorkReporter:
    """Deterministic fallback for Prompt 13 (Prior-Work Evidence Reporter)."""

    @classmethod
    def report(cls, hypothesis: str, results: Any) -> PriorWorkEvidenceOutput:
        items: List[Dict[str, Any]] = []
        if isinstance(results, list):
            items = [r for r in results if isinstance(r, dict)]
        elif isinstance(results, str) and results.strip():
            try:
                loaded = json.loads(results)
                if isinstance(loaded, list):
                    items = [r for r in loaded if isinstance(r, dict)]
            except Exception:
                pass

        n = len(items)
        hyp_low = hypothesis.lower()
        hyp_archs = [a.lower() for a in KNOWN_ARCHITECTURES if a.lower() in hyp_low]
        hyp_datasets = [d.lower() for d in KNOWN_DATASETS if d.lower() in hyp_low]
        hyp_metrics = [m.lower() for m in KNOWN_METRICS if m.lower() in hyp_low]

        scored: List[Tuple[int, ClosestPaperItem]] = []
        for idx, item in enumerate(items, start=1):
            pid = str(item.get("id") or item.get("paper_id") or f"R-{idx}").strip()
            title = str(item.get("title") or "Untitled").strip()
            abstract = str(item.get("abstract") or item.get("summary") or "").strip()
            year = item.get("year", "")
            text_low = f"{title} {abstract}".lower()

            overlap: List[str] = []
            if any(a in text_low for a in hyp_archs):
                overlap.append("architecture")
            if any(d in text_low for d in hyp_datasets):
                overlap.append("dataset")
            if any(m in text_low for m in hyp_metrics):
                overlap.append("metric")

            score = len(overlap)
            if not overlap:
                overlap = ["none"]
                note = f"No architecture, dataset, or metric overlap found in abstract ({year or 'year n/a'})."
            else:
                note = f"Abstract mentions matching {', '.join(overlap)} ({year or 'year n/a'})."

            scored.append(
                (
                    score,
                    ClosestPaperItem(
                        id=pid,
                        title=title,
                        overlap=overlap,  # type: ignore
                        note=note,
                    ),
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [c for _, c in scored[:5]]
        any_overlap = any(s > 0 for s, _ in scored[:5])

        if not any_overlap:
            statement = f"No close match among {n} results searched; manual check required."
        else:
            matched_cnt = sum(1 for s, _ in scored if s > 0)
            statement = (
                f"Found {matched_cnt} paper(s) with partial overlap among {n} results searched; "
                "manual review of full text is recommended."
            )

        return PriorWorkEvidenceOutput(closest=top_items, statement=statement)


class RuleBasedSpokenExplainer:
    """Deterministic fallback for Prompt 14 (Spoken Explanation Formatter)."""

    @classmethod
    def format(cls, result_json: Any) -> SpokenExplanationOutput:
        data: Dict[str, Any] = {}
        if isinstance(result_json, dict):
            data = result_json
        elif isinstance(result_json, str) and result_json.strip():
            try:
                loaded = json.loads(result_json)
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {"summary": result_json}

        # Collect sources ("P-... p.N")
        sources: List[str] = []
        raw_sources = data.get("sources")
        if isinstance(raw_sources, list):
            for s in raw_sources:
                s_str = str(s).strip()
                if s_str and s_str not in sources:
                    sources.append(s_str)

        if not sources:
            # Extract from paper_a / paper_b or evidence_ids or regex
            pa = data.get("paper_a") or data.get("paper_a_id")
            pb = data.get("paper_b") or data.get("paper_b_id")
            if pa:
                pg_a = data.get("page_a", 1)
                sources.append(f"{pa} p.{pg_a}")
            if pb:
                pg_b = data.get("page_b", 1)
                sources.append(f"{pb} p.{pg_b}")

        if not sources:
            for key in ("evidence_ids", "papers_to_read", "cited_paper_ids", "motivating_papers"):
                vals = data.get(key)
                if isinstance(vals, list):
                    for pid in vals:
                        tag = f"{pid} p.1"
                        if tag not in sources:
                            sources.append(tag)

        if not sources:
            found_ids = re.findall(r"\bP-[A-Z0-9]+\b", json.dumps(data))
            for pid in found_ids[:4]:
                tag = f"{pid} p.1"
                if tag not in sources:
                    sources.append(tag)

        if not sources:
            sources = ["P-CORPUS p.1"]

        # Build plain-language explanation from available fields
        parts: List[str] = []
        title = data.get("title") or data.get("gap_label") or ""
        verdict = data.get("verdict") or data.get("classification") or ""
        reason = data.get("explanation") or data.get("reason") or data.get("statement") or data.get("summary") or ""

        if title:
            parts.append(f"Here is a candidate option: {title}.")
        if verdict:
            parts.append(f"The comparison status is {str(verdict).replace('_', ' ')}.")
        if reason:
            parts.append(str(reason).strip())
        if not parts:
            parts.append("Here is a candidate summary from the loaded papers.")

        # Strip markdown symbols and replace 'finding' with 'candidate'
        text = " ".join(parts)
        for sym in ("**", "##", "```", "|", "*", "`"):
            text = text.replace(sym, "")
        text = re.sub(r"\bfindings\b", "candidates", text, flags=re.I)
        text = re.sub(r"\bfinding\b", "candidate", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()

        source_clause = f"You can check {', '.join(sources[:4])} on screen."
        max_body_words = max(10, 118 - count_words(source_clause))
        body_words = text.split()[:max_body_words]
        body_text = " ".join(body_words).rstrip(".,;:") + "."

        spoken_text = f"{body_text} {source_clause}".strip()
        return SpokenExplanationOutput(spoken_text=spoken_text, sources=sources[:6])


class RuleBasedGroundedQA:
    """Deterministic fallback for Prompt 15 (Grounded JSON Passage Q&A)."""

    @staticmethod
    def _parse_passages(passages: Any) -> List[Dict[str, Any]]:
        if isinstance(passages, list):
            return [p for p in passages if isinstance(p, dict)]
        if isinstance(passages, str) and passages.strip():
            try:
                loaded = json.loads(passages)
                if isinstance(loaded, list):
                    return [p for p in loaded if isinstance(p, dict)]
            except Exception:
                pass
            # Parse formatted blocks like "[1] (P-123, Page 4): text"
            blocks: List[Dict[str, Any]] = []
            for m in re.finditer(r"\[(\d+)\]\s*\(([^,]+),\s*Page\s*(\d+)\):\s*(.+?)(?=\n\[\d+\]|\Z)", passages, re.S | re.I):
                blocks.append({
                    "id": m.group(2).strip(),
                    "page": int(m.group(3)),
                    "text": m.group(4).strip(),
                })
            return blocks
        return []

    @classmethod
    def answer(cls, standalone_question: str, passages: Any) -> GroundedQAOutput:
        items = cls._parse_passages(passages)
        q_tokens = {
            t.lower()
            for t in re.findall(r"\w+", standalone_question)
            if len(t) >= 3 and t.lower() not in {"what", "which", "where", "when", "does", "that", "this", "with", "from", "the", "and", "for", "are", "how"}
        }

        scored: List[Tuple[int, str, int, str]] = []
        for item in items:
            pid = str(item.get("paper_id") or item.get("id") or "P-CORPUS").strip()
            try:
                page = max(0, int(item.get("page", 1)))
            except Exception:
                page = 1
            text = str(item.get("text") or "").strip()
            for sent in re.split(r"(?<=[.!?])\s+", text):
                sent_clean = sent.strip()
                if len(sent_clean) < 10:
                    continue
                s_low = sent_clean.lower()
                hits = sum(1 for tok in q_tokens if tok in s_low)
                if hits > 0:
                    scored.append((hits, pid, page, sent_clean))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            missing_terms = ", ".join(sorted(q_tokens)[:5]) if q_tokens else "the requested topic"
            return GroundedQAOutput(
                answerable=False,
                answer=f"The provided passages do not answer this question; missing direct evidence about {missing_terms}.",
                claims=[],
                follow_ups=[
                    "Which specific paper or dataset should be searched next?",
                    "Would you like to expand the search to arXiv or OpenAlex?",
                ],
            )

        claims: List[GroundedClaimItem] = []
        ans_parts: List[str] = []
        seen_quotes = set()
        for _, pid, page, sent_clean in scored[:3]:
            quote = " ".join(sent_clean.split()[:25])
            if quote in seen_quotes:
                continue
            seen_quotes.add(quote)
            claims.append(
                GroundedClaimItem(
                    text=sent_clean,
                    paper_id=pid,
                    page=page,
                    quote=quote,
                )
            )
            ans_parts.append(f"{sent_clean.rstrip('.')} [{pid}, p.{page}].")

        return GroundedQAOutput(
            answerable=True,
            answer=" ".join(ans_parts),
            claims=claims,
            follow_ups=[
                "What experimental split was used for this result?",
                "Do other papers in the corpus report different numbers?",
            ],
        )
