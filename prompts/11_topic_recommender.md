---
name: "11 Topic recommender"
file: "prompts/11_topic_recommender.md"
version: "1.0"
stage: "topic_recommend"
purpose: "recommend 3 to 5 evidence-traceable student project topics from the Gap Matrix and contradictions using only provided evidence."
---

# 11 Topic recommender

File: prompts/11_topic_recommender.md | Version 1.0  
Used by: Topic & Scope Advisor (one call after scope criteria / interview answers are collected, only when a language model is configured)  
Purpose: recommend 3 to 5 mini or major project options strictly traceable to gap cells or contradictions.

## System prompt

You recommend student project topics using ONLY the evidence provided. Output JSON only.

## User prompt (template)

INPUT:
- User profile: {{profile}}
- Gap matrix cells (Architecture x Dataset, with status and supporting paper IDs): {{gaps}}
- Contradictions found: {{contradictions}}
- Scope criteria: {{scope_criteria}}

TASK: Propose 3 to 5 options. Each must be traceable to at least one gap cell or contradiction.

RULES:
- Never use the words "novel", "first", "unexplored", or "state of the art".
- Use the wording "not found in the {{n}} papers searched".
- Scope must follow the provided criteria exactly.
- Do not invent datasets, papers, or numbers.
- If the evidence is too thin for 3 options, return fewer and say why.

OUTPUT:
{"options":[{"title":"","scope":"mini|major","reason":"","evidence_ids":["P-..."],"datasets":[""],"compute_needed":"","papers_to_read":["P-..."],"risks":[""],"effort_weeks":0}],"warnings":[""]}
