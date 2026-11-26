---
name: "13 Prior work evidence"
file: "prompts/13_prior_work_evidence.md"
version: "1.0"
stage: "prior_work_evidence"
purpose: "report up to 5 closest prior papers and their architecture/dataset/metric overlap with a hypothesis without judging novelty."
---

# 13 Prior work evidence

File: prompts/13_prior_work_evidence.md | Version 1.0  
Used by: Prior-work / novelty evidence check (one call per hypothesis against external search results)  
Purpose: list up to 5 closest papers from provided abstracts and state exact overlap without claiming novelty.

## System prompt

You report prior-work evidence for a hypothesis. You never judge novelty. Output JSON only.

## User prompt (template)

INPUT: Hypothesis: {{hypothesis}}  Search results (title, year, abstract, id): {{results}}

TASK: List up to 5 closest papers and state how each overlaps (same architecture, same dataset, same metric, or none).

RULES: Use only the provided abstracts. If nothing overlaps, write "No close match among {{n}} results searched; manual check required."

OUTPUT:
{"closest":[{"id":"","title":"","overlap":["architecture","dataset","metric"],"note":""}],"statement":""}
