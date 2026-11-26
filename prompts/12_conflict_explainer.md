---
name: "12 Conflict explainer"
file: "prompts/12_conflict_explainer.md"
version: "1.0"
stage: "conflict_explain"
purpose: "judge whether two paper results truly conflict or reflect condition mismatches, with verbatim quotes and page numbers."
---

# 12 Conflict explainer

File: prompts/12_conflict_explainer.md | Version 1.0  
Used by: Cross-Paper Contradiction Judgement (pair-level condition comparison)  
Purpose: distinguish true conflicts from dataset/metric/split/hyperparameter/seed/architecture mismatches with verbatim quotes <= 25 words and page numbers.

## System prompt

You judge whether two paper results truly conflict. Output JSON only.

## User prompt (template)

INPUT: Record A: {{record_a}}  Record B: {{record_b}}
(each has dataset, metric, split, architecture, hyperparameters, value, verbatim quote, page)

TASK: Compare the conditions. Decide:
- "true_conflict" only if dataset, metric, and split all match and the conclusions differ.
- "condition_mismatch" if any of them differ; list the differing factors.
- "insufficient_info" if a needed field is missing.

RULES: Quote each side verbatim (25 words or fewer) with page number. Do not guess missing fields.

OUTPUT:
{"verdict":"", "differing_factors":["dataset|metric|split|hyperparameters|seed|architecture"], "quote_a":"", "page_a":0, "quote_b":"", "page_b":0, "confidence":"low|medium|high", "explanation":""}
