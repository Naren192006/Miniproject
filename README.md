# HypothesisForge: Validation-First, Scope-Aware, Evidence-Traceable Hypotheses for ML Projects

**Core Claim:** *Validation-first, scope-aware, evidence-traceable hypotheses for ML projects — with measured grounding precision, validator ablations, and executable small-scale verification.*

This repository implements the versioned prompt loading, templating, execution, validation, evaluation harness, and deterministic fallback system for **HypothesisForge**.

## Measured Evaluation Harness (16 Hand-Labeled Papers)

Run `python scripts/run_grounding_benchmark.py` to reproduce the grounding, gap-matrix, contradiction-precision, and validator ablation benchmark:

| System Variant | Precision | Recall | Unsupported Claim Rate | Quote Pass Rate | Gap Matrix Precision | Contradiction Precision |
|---|---:|---:|---:|---:|---:|---:|
| **Plain Unvalidated Baseline** | 60.0% | 75.0% | 40.0% | 60.0% | 50.0% | 0.0% |
| **Ablation (No Split/Noise Filter)** | 92.3% | 100.0% | 7.7% | 100.0% | 100.0% | 0.0% |
| **HypothesisForge (Validated)** | **100.0%** | **100.0%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** |

## Honest Framing & Known Limits
- **No novelty claims:** Gap cells and prior-work checks report *"not found in the N papers searched"* or *"No close match among N results searched; manual check required."* — never *"novel"* or *"first"*.
- **Condition-aware contradictions:** Differences across papers are decomposed into `dataset`, `metric`, `split`, `hyperparameters`, `seed`, and `architecture` before flagging a `true_conflict`.
- **Small-scale execution:** Automated experiment stubs run on lightweight benchmarks (`sklearn.datasets.load_digits` with OOD noise/contrast shift) to verify the hypothesis-to-evidence loop; large-scale GPU benchmarks remain a manual step.

## Completed Prompts (01 – 15)

### 1. `01 Literature Extraction` (`prompts/01_literature_extraction.md`)
- **Version:** 1.1
- **Stage:** `literature` (invoked once per ingested research paper)
- **Purpose:** Extracts structured experimental records directly from paper text (tasks, architectures, datasets, training strategies, shift settings, metrics, evaluation protocol, numeric results, findings, and limitations).
- **Core Rules Enforced:**
  - Strict JSON output without markdown fences or extra commentary.
  - Verbatim quotes of at most 25 words for all reported results, findings, and limitations.
  - Max 8 reported results, max 6 findings, max 6 limitations.
  - Missing details must remain `null` or empty lists (no hallucination).
  - Page numbers must come from the source block.

### 2. `02 Gap Relevance` (`prompts/02_gap_relevance.md`)
- **Version:** 1.1
- **Stage:** `gaps` (invoked once per candidate gap against corpus experimental matrix)
- **Purpose:** Assesses the scientific relevance of an untested experimental combination within the corpus context.
- **Core Rules Enforced:**
  - Never state that the candidate gap is "novel" or that "nobody has tested it" (gaps are treated strictly as candidate hypotheses).
  - If the corpus evidence is too thin to judge, rating must be set to `low` and reason must state the evidence is thin.
  - Absence explanation must be one of: `"corpus size"`, `"impossible or meaningless pairing"`, `"not reported by these papers but plausible"`, or `"unknown"`.
  - Max 3 verification questions.

### 3. `03 Contradiction Judgement` (`prompts/03_contradiction_judgement.md`)
- **Version:** 1.1
- **Stage:** `contradictions` (invoked once per compared pair of experimental results)
- **Purpose:** Determines whether two reported results truly conflict or whether differences in datasets, protocols, backbones, or splits explain the discrepancy.
- **Core Rules Enforced:**
  - Never declare either paper wrong.
  - If protocol, metric definition, or split is not stated on either side, classification must be `"insufficient information"` or `"non-comparable evidence"`, never `"apparent disagreement"`.
  - Max 4 conditions to verify in the original papers.

### 4. `04 Hypothesis Generation` (`prompts/04_hypothesis_generation.md`)
- **Version:** 1.1
- **Stage:** `hypotheses` (invoked once per candidate gap)
- **Purpose:** Formulates one precise, testable, falsifiable hypothesis grounded strictly in provided evidence, or formally declines when evidence is insufficient.
- **Core Rules Enforced:**
  - Strict statement syntax: `"If ..., then ..., measured by ..., relative to ..."`.
  - Both sides of the gap named exactly.
  - Vague verbs such as `"improve"` are prohibited unless the metric is explicitly named in the statement.
  - No claims of novelty or claims unsupported by cited evidence.
  - Cites strictly paper IDs from provided evidence (must not be empty when decision is `"hypothesis"`).
  - Max 4 assumptions.
  - Declining with `decision: "insufficient_evidence"` is a valid first-class outcome.

### 5. `05 Scientific Critic` (`prompts/05_scientific_critic.md`)
- **Version:** 1.1
- **Stage:** `critic` (invoked once per proposed hypothesis)
- **Purpose:** Strict scientific criticism seeking weak evidence, unfalsifiable claims, confounders, and literature overlap.
- **Core Rules Enforced:**
  - Choose `"reject"` when the claim cannot be traced to the evidence.
  - Choose `"revise"` when the claim is plausible but not specific enough, when prior work may overlap, or when the prior-work search was not run.
  - Choose `"accept"` only when the claim is specific, falsifiable, measurable, and traceable, the prior-work search was run, and no listed prior work covers it.
  - Prohibits claims of global novelty.
  - Max 4 missing evidence items, max 4 alternative explanations (confounders).

### 6. `06 Experiment Design` (`prompts/06_experiment_design.md`)
- **Version:** 1.1
- **Stage:** `experiments` (invoked once per accepted/revised hypothesis)
- **Purpose:** Refines a draft experiment plan into a fair, reproducible, and feasible protocol with grounded datasets.
- **Core Rules Enforced:**
  - If the dataset is not in the grounded list, must begin with `"SUGGESTION, VERIFY:"` and explain why it fits.
  - Any baseline not in the grounded list must be marked with `"(verify)"`.
  - Procedure list limited to at most 7 ordered steps.
  - Must state at least 3 random seeds.
  - Prohibits stating expected outcomes as facts.

### 7. `07 Report Summary` (`prompts/07_report_summary.md`)
- **Version:** 1.0
- **Stage:** `report` (invoked once per run)
- **Purpose:** Plain-language summary stating the scope and limits of the analysis.
- **Core Rules Enforced:**
  - Strict plain text response (not JSON).
  - No more than 180 words in exactly three short paragraphs.
  - No bullet points and no headings.
  - Do not claim a gap is "new", "unexplored", or a "discovery".
  - Mentions unrun analyses and declined hypotheses.

### 8. `08 Evidence Answer` (`prompts/08_evidence_answer.md`)
- **Version:** 1.1
- **Stage:** `ask_the_papers` (invoked once per user question)
- **Purpose:** Answers questions strictly using numbered passages with verifiable citations.
- **Core Rules Enforced:**
  - Strict plain text response.
  - 1 to 4 short paragraphs.
  - Every claim must cite a passage number in square brackets (e.g. `[1]`).
  - Only uses numbers that appear in the passage list (rejects hallucinations/unlisted numbers).
  - If passages do not contain the answer, strictly outputs: `"The passages do not answer this question."`.

### 9. `09 Query Rewriter` (`prompts/09_query_rewriter.md`)
- **Stage:** `query_rewrite` — Rewrites follow-up questions into standalone retrieval queries, classifies intent (`summary | compare | gap | contradiction | topic_advice | factual_qa | out_of_scope`), and splits multi-part questions into at most 3 sub-queries.

### 10. `10 Topic Interviewer` (`prompts/10_topic_interviewer.md`)
- **Stage:** `topic_interview` — Asks one short plain-language question at a time until all 6 student profile fields (`field_of_interest`, `current_skills`, `available_weeks`, `compute_access`, `project_level`, `preferred_outcome`) are collected.

### 11. `11 Topic Recommender` (`prompts/11_topic_recommender.md`)
- **Stage:** `topic_recommend` — Proposes 3–5 mini or major student project options strictly traceable to Gap Matrix cells or contradictions, forbidding `"novel"`, `"first"`, `"unexplored"`, and `"state of the art"`.

### 12. `12 Conflict Explainer` (`prompts/12_conflict_explainer.md`)
- **Stage:** `conflict_explain` — Distinguishes `true_conflict` from `condition_mismatch` and `insufficient_info` with verbatim quotes (`<= 25` words) and page numbers.

### 13. `13 Prior Work Evidence` (`prompts/13_prior_work_evidence.md`)
- **Stage:** `prior_work_evidence` — Reports up to 5 closest papers and their `architecture` / `dataset` / `metric` overlap without judging novelty.

### 14. `14 Spoken Explainer` (`prompts/14_spoken_explainer.md`)
- **Stage:** `spoken_explain` — Converts structured results into a `<= 120`-word plain-language spoken summary using `"candidate"` instead of `"finding"` and naming paper IDs and pages.

### 15. `15 Grounded QA` (`prompts/15_grounded_qa.md`)
- **Stage:** `grounded_qa` — Answers standalone questions strictly from provided passages with `[paper_id, page]` citations and `<= 25`-word verbatim quotes, or returns `answerable: false`.

---

## System Architecture & Recovery Workflow

As defined in the Prompt Pack specification:
1. **Dynamic Templating:** Prompts are versioned Markdown files loaded from `prompts/`. Double-brace placeholders (`{{...}}`) are validated and substituted at runtime.
2. **Schema & Rule Validation:** Replies are validated using strict Pydantic models with custom validators for quote lengths, banned vocabulary (e.g. novelty claims), and bounds.
3. **One-Attempt Retry:** If the LLM generates an invalid or non-compliant response, an explicit correction prompt is generated and retried once.
4. **Deterministic Fallback:** If the retry fails or no LLM is configured, the system falls back to a deterministic rule-based extractor/evaluator.

---

## Verification & Reproducibility

Install dependencies:
```bash
pip install -e .
```

Run all unit and validation tests (`100` tests):
```bash
python -m pytest -q
```

Run the 16-paper grounding benchmark & validator ablation:
```bash
python scripts/run_grounding_benchmark.py
```

Launch the Streamlit application:
```bash
streamlit run app.py
```
