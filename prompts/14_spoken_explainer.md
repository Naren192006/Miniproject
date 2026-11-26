---
name: "14 Spoken explainer"
file: "prompts/14_spoken_explainer.md"
version: "1.0"
stage: "spoken_explain"
purpose: "turn structured analysis results into a plain-language spoken explanation (max 120 words) citing paper IDs and pages."
---

# 14 Spoken explainer

File: prompts/14_spoken_explainer.md | Version 1.0  
Used by: Voice / spoken summary layer (one call per spoken readout, only when a language model is configured)  
Purpose: convert structured JSON results into a short, non-technical spoken explanation ending with paper IDs and pages to check on screen.

## System prompt

Turn the structured result below into a spoken explanation for a non-technical listener. Output JSON only.

## User prompt (template)

INPUT: {{result_json}}

RULES:
- Max 120 words, short sentences, no symbols, tables, or jargon (explain any technical term in a few words).
- Say "candidate" instead of "finding" for gaps and contradictions.
- End by naming the paper IDs and pages the user can check on screen.
- Do not add anything not in the input.

OUTPUT:
{"spoken_text":"", "sources":["P-... p.N"]}
