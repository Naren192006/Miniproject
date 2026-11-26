---
name: "15 Grounded qa"
file: "prompts/15_grounded_qa.md"
version: "1.0"
stage: "grounded_qa"
purpose: "answer standalone retrieval questions using only provided passages, returning structured claims with paper_id, page, and <= 25-word verbatim quotes."
---

# 15 Grounded qa

File: prompts/15_grounded_qa.md | Version 1.0  
Used by: "Ask the papers" structured Q&A (one call per standalone question, only when a language model is configured)  
Purpose: answer strictly from retrieved passages with per-claim [paper_id, page] citations and <= 25-word verbatim quotes, or report answerable: false.

## System prompt

Answer using ONLY the passages below. Output JSON only.

## User prompt (template)

INPUT: Question: {{standalone_question}}  Passages (id, page, text): {{passages}}

RULES:
- Every claim must cite [paper_id, page] and include a verbatim quote of 25 words or fewer.
- If the passages do not answer it, return "answerable": false and say what is missing.
- No outside knowledge.

OUTPUT:
{"answerable":true, "answer":"", "claims":[{"text":"","paper_id":"","page":0,"quote":""}], "follow_ups":["","",""]}
