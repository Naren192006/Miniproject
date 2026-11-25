# HypothesisForge: Prompt Pack & Verification System

This repository implements the versioned prompt loading, templating, execution, validation, and fallback system for **HypothesisForge** (Multi-Agent Research Gap Discovery and Experimental Hypothesis Generation System).

## Completed Prompts

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

---

## System Architecture & Recovery Workflow

As defined in the Prompt Pack specification:
1. **Dynamic Templating:** Prompts are versioned Markdown files loaded from `prompts/`. Double-brace placeholders (`{{...}}`) are validated and substituted at runtime.
2. **Schema & Rule Validation:** Replies are validated using strict Pydantic models with custom validators for quote lengths, banned vocabulary (e.g. novelty claims), and bounds.
3. **One-Attempt Retry:** If the LLM generates an invalid or non-compliant response, an explicit correction prompt is generated and retried once.
4. **Deterministic Fallback:** If the retry fails or no LLM is configured, the system falls back to a deterministic rule-based extractor/evaluator.

---

## Verification & Testing

Run all unit and validation tests:
```bash
python -m pytest -v
```

Run the end-to-end verification audit:
```bash
python scripts/verify_prompts.py
```
