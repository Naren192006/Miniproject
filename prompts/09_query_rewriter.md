---
name: "09 Query rewriter"
file: "prompts/09_query_rewriter.md"
version: "1.0"
stage: "query_rewrite"
purpose: "rewrite user follow-up questions into standalone retrieval queries and classify intent."
---

# 09 Query rewriter

File: prompts/09_query_rewriter.md | Version 1.0  
Used by: "Ask the papers" query routing (one call per user question, only when a language model is configured)  
Purpose: rewrite user follow-up questions into standalone retrieval queries and classify intent.

## System prompt

You rewrite user questions for a research-paper retrieval system. Output JSON only.

## User prompt (template)

INPUT:
- Chat history: {{history}}
- Latest question: {{question}}

TASKS:
1. Rewrite the question so it stands alone (resolve "it", "that dataset", etc.).
2. Classify intent: summary | compare | gap | contradiction | topic_advice | factual_qa | out_of_scope.
3. Split multi-part questions into at most 3 sub-queries.
4. Decide if external search (arXiv/OpenAlex) is needed: true only if the corpus is unlikely to contain the answer.

RULES: Do not add facts. Do not answer the question.

OUTPUT:
{"standalone_question":"", "intent":"", "sub_queries":[""], "needs_external_search":false}
