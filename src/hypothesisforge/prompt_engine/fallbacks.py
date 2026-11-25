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

                # Require a numeric value (either % or decimal/number paired with metric) in the same sentence
                num_match = re.search(r"\b(\d{1,4}(?:\.\d{1,3})?)\s*%", sent_clean)
                if not num_match:
                    # Also match decimal or score numbers near the metric/result verb (e.g., "RMSE of 1.45", "AUC of 0.94")
                    metric_esc = re.escape(metric_match)
                    num_match = re.search(
                        rf"(?:{metric_esc}|achieves|reaches|obtains|reduces|improves|of|by)\s+(\d+(?:\.\d+)?)\b",
                        sent_clean,
                        re.I,
                    )
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
        values = re.findall(r"-\s*Value:\s*([^\n]+)", pair_text)
        p_a = papers[0].strip() if len(papers) > 0 else "Paper A"
        p_b = papers[1].strip() if len(papers) > 1 else "Paper B"
        d_a = datasets[0].strip() if len(datasets) > 0 else "not stated"
        d_b = datasets[1].strip() if len(datasets) > 1 else "not stated"
        v_a = values[0].strip() if len(values) > 0 else "a reported value"
        v_b = values[1].strip() if len(values) > 1 else "a reported value"

        known_found = [d for d in KNOWN_DATASETS if d.lower() in pair_lower]
        if len(set(known_found)) > 1 or (
            d_a != d_b
            and "unknown" not in d_a.lower()
            and "not stated" not in d_a.lower()
            and "unknown" not in d_b.lower()
            and "not stated" not in d_b.lower()
        ):
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
        }
        question_words = {
            w for w in re.findall(r"[a-zA-Z0-9\-]{3,}", question.lower())
            if w not in stop_words
        }
        if not question_words:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        # Score individual sentences inside each passage so we return the sentence that actually answers the question
        scored_sentences: List[Tuple[int, int, str]] = []
        for p_num_str, p_text in passage_blocks:
            p_num = int(p_num_str)
            cleaned = re.sub(r"^\([^)]+\):\s*", "", p_text).strip()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue
                s_low = sent_clean.lower()
                hits = sum(1 for w in question_words if re.search(r"\b" + re.escape(w) + r"\b", s_low))
                if hits > 0:
                    scored_sentences.append((hits, p_num, sent_clean))

        if not scored_sentences:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        # Require meaningful match (at least 2 keyword hits, or 1 hit if question only had 1 keyword)
        min_hits = 2 if len(question_words) >= 2 else 1
        top_sentences = [item for item in scored_sentences if item[0] >= min_hits][:3]
        if not top_sentences:
            text = "The passages do not answer this question."
            return EvidenceAnswerOutput(text=text, paragraphs=[text], citations=[], is_declined=True)

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
