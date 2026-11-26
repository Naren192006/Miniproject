"""
HypothesisForge - Response Validator
Validates model replies for Prompts 01 - 08 against Prompt Pack rules and Pydantic schemas.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, ValidationError

from .models import (
    LiteratureExtractionOutput,
    GapRelevanceOutput,
    ContradictionJudgementOutput,
    HypothesisGenerationOutput,
    ScientificCriticOutput,
    ExperimentDesignOutput,
    ReportSummaryOutput,
    EvidenceAnswerOutput,
    QueryRewriteOutput,
    TopicInterviewOutput,
    TopicRecommendationOutput,
    ConflictExplainerOutput,
    PriorWorkEvidenceOutput,
    SpokenExplanationOutput,
    GroundedQAOutput,
    count_words,
)


@dataclass
class ValidationResult:
    is_valid: bool
    data: Optional[BaseModel] = None
    raw_json: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    error_message: Optional[str] = None
    retry_prompt: Optional[str] = None


class ResponseValidator:
    """
    Validates model replies across all Prompt Pack stages.
    """

    @staticmethod
    def clean_json_text(text: str) -> str:
        """Strips markdown code blocks or commentary around JSON."""
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    @staticmethod
    def parse_json_object(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Ensures the reply is a single JSON object (dict)."""
        cleaned = ResponseValidator.clean_json_text(raw_text)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return None, f"Response is not valid JSON: {str(e)}"

        if not isinstance(parsed, dict):
            return None, f"Response must be a single JSON object (got {type(parsed).__name__})"

        return parsed, None

    @classmethod
    def validate_literature_extraction(cls, raw_text: str, source_text: Optional[str] = None) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return exactly one valid JSON object and nothing else."
            )

        try:
            output = LiteratureExtractionOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nPlease correct the JSON format and follow all rules exactly."
            )

        if source_text:
            unfound_quotes = []
            for idx, res in enumerate(output.reported_results):
                if res.quote and res.quote not in source_text:
                    unfound_quotes.append(f"Result {idx+1} quote not found in source text: '{res.quote}'")
                elif res.quote:
                    # Verify the reported numeric value actually appears in the quote
                    val_str = str(res.value)
                    val_int_str = str(int(res.value)) if isinstance(res.value, float) and res.value.is_integer() else val_str
                    if val_str not in res.quote and val_int_str not in res.quote:
                        unfound_quotes.append(f"Result {idx+1} value '{res.value}' does not appear in its quote '{res.quote}'")
            for idx, find in enumerate(output.findings):
                if find.quote and find.quote not in source_text:
                    unfound_quotes.append(f"Finding {idx+1} quote not found in source text: '{find.quote}'")
            for idx, lim in enumerate(output.limitations):
                if lim.quote and lim.quote not in source_text:
                    unfound_quotes.append(f"Limitation {idx+1} quote not found in source text: '{lim.quote}'")

            if unfound_quotes:
                msg = "Rule violation: Quotes must be copied exactly from the paper text. " + "; ".join(unfound_quotes)
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nProvide verbatim quotes only, or drop the claim if not directly in the text."
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_gap_relevance(cls, raw_text: str) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return a single JSON object."
            )

        try:
            output = GapRelevanceOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nRemember: Never claim a gap is novel or that nobody has tested it. Return a single JSON object."
            )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_contradiction_judgement(cls, raw_text: str, pair_context: Optional[str] = None) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return a single JSON object."
            )

        try:
            output = ContradictionJudgementOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nRemember: Never declare either paper wrong. Return a single JSON object."
            )

        if pair_context:
            context_lower = pair_context.lower()
            unspecified_protocol = (
                "protocol: not stated" in context_lower
                or "protocol: \"not stated\"" in context_lower
                or "split: not stated" in context_lower
                or "split is not stated" in context_lower
                or "metric: not stated" in context_lower
            )
            if unspecified_protocol and output.classification == "apparent disagreement under comparable settings":
                msg = (
                    "Rule violation: If the protocol, metric definition or split is not stated on either side, "
                    "choose 'insufficient information' or 'non-comparable evidence', never 'apparent disagreement'."
                )
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nPlease re-classify as 'insufficient information' or 'non-comparable evidence'."
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_hypothesis_generation(cls, raw_text: str, evidence_context: Optional[str] = None) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return a single JSON object."
            )

        try:
            output = HypothesisGenerationOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nRemember: Name both sides of the gap, cite only paper IDs from evidence, do not claim novelty, and avoid vague verbs such as 'improve'."
            )

        if evidence_context and output.decision == "hypothesis" and output.cited_paper_ids:
            unfound_ids = [pid for pid in output.cited_paper_ids if pid not in evidence_context]
            if unfound_ids:
                msg = f"Rule violation: Citing paper IDs that do not appear in the evidence: {unfound_ids}"
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nCite only paper IDs that are explicitly present in the evidence."
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_scientific_critic(
        cls,
        raw_text: str,
        prior_status: Optional[str] = None,
        overlap: Optional[str] = None,
        checks: Optional[str] = None,
    ) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return a single JSON object."
            )

        try:
            output = ScientificCriticOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nPlease follow all critic verdict rules and return a single JSON object."
            )

        if checks:
            checks_lower = checks.lower()
            if ("cannot be traced" in checks_lower or "traceability: failed" in checks_lower or "unsupported" in checks_lower):
                if output.verdict != "reject":
                    msg = "Verdict rule violation: Choose 'reject' when the claim cannot be traced to the evidence."
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nThe automated checks indicate claim is untraced. Change verdict to 'reject'."
                    )

        if prior_status:
            status_lower = prior_status.lower().strip()
            if status_lower in ["not run", "not_run", "skipped", "unrun", "pending"]:
                if output.verdict == "accept":
                    msg = "Verdict rule violation: Choose 'revise' when the prior-work search was not run."
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nPrior work search was not run, so verdict cannot be 'accept'. Change verdict to 'revise'."
                    )

        if overlap:
            overlap_clean = overlap.strip()
            if overlap_clean and not any(neg in overlap_clean.lower() for neg in ["none", "no prior work", "empty", "no overlap", "[]"]):
                if output.verdict == "accept":
                    msg = "Verdict rule violation: Choose 'accept' only when no listed prior work covers it. Prior work was found, so choose 'revise' or 'reject'."
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nPrior work titles overlap with the claim. Re-evaluate verdict to 'revise'."
                    )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_experiment_design(
        cls,
        raw_text: str,
        grounded_options: Optional[str] = None,
    ) -> ValidationResult:
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Please return a single JSON object."
            )

        try:
            output = ExperimentDesignOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nPlease fix the protocol fields and follow all rules exactly."
            )

        if grounded_options:
            grounded_lower = grounded_options.lower()
            ds_val = output.dataset_and_split
            ds_is_grounded = any(
                term.lower() in grounded_lower for term in [
                    "pacs", "officehome", "office-home", "vlcs", "domainnet", "terraincognita", "camelyon17"
                ] if term.lower() in ds_val.lower()
            )
            if not ds_is_grounded and not ds_val.startswith("SUGGESTION, VERIFY:"):
                msg = (
                    "Rule violation: If the dataset is not in the grounded list, "
                    "begin with 'SUGGESTION, VERIFY:' and say why it fits."
                )
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nUpdate 'dataset_and_split' to start with 'SUGGESTION, VERIFY:'."
                )

            unmarked_baselines = []
            for b in output.baselines:
                b_name = b.replace("(verify)", "").strip().lower()
                if b_name not in grounded_lower and "(verify)" not in b:
                    unmarked_baselines.append(b)
            if unmarked_baselines:
                msg = f"Rule violation: Mark any baseline not in the grounded list with '(verify)': {unmarked_baselines}"
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nAppend '(verify)' to baselines that do not appear in the grounded options."
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_report_summary(cls, raw_text: str, facts_json: Optional[str] = None) -> ValidationResult:
        """
        Validates Prompt 07 (07_report_summary.md) output.
        Plain text of no more than 180 words in three short paragraphs.
        """
        text = raw_text.strip()

        # Split into paragraphs
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        total_words = count_words(text)

        try:
            output = ReportSummaryOutput(
                text=text,
                paragraphs=raw_paragraphs,
                word_count=total_words
            )
        except ValidationError as ve:
            error_details = [f"{err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_text=text,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nPlease write a summary of no more than 180 words in three short paragraphs with no bullets or headings."
            )

        return ValidationResult(is_valid=True, data=output, raw_text=text)

    @classmethod
    def validate_query_rewrite(cls, raw_text: str) -> ValidationResult:
        """Validates Prompt 09 (09_query_rewriter.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the required keys.",
            )

        try:
            output = QueryRewriteOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nReturn valid JSON with at most 3 sub_queries.",
            )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_evidence_answer(cls, raw_text: str, passages: Optional[str] = None) -> ValidationResult:
        """
        Validates Prompt 08 (08_evidence_answer.md) output.
        Plain text in one to four short paragraphs with numbered citations [n].
        """
        text = raw_text.strip()
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Extract citation numbers [n]
        found_citations = [int(c) for c in re.findall(r"\[(\d+)\]", text)]

        try:
            output = EvidenceAnswerOutput(
                text=text,
                paragraphs=raw_paragraphs,
                citations=found_citations
            )
        except ValidationError as ve:
            error_details = [f"{err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_text=text,
                error_message=msg,
                retry_prompt=f"Validation errors encountered:\n{msg}\nPlease answer in one to four short paragraphs citing passages in square brackets."
            )

        # Rule: Only use numbers that appear in the passage list
        if passages and not output.is_declined:
            available_passage_numbers = set(int(n) for n in re.findall(r"\[(?:Passage\s*)?(\d+)\]", passages))
            invalid_citations = [c for c in found_citations if c not in available_passage_numbers]
            if invalid_citations:
                msg = f"Rule violation: Only use numbers that appear in the passage list. Invalid citations: {invalid_citations}"
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_text=text,
                    error_message=msg,
                    retry_prompt=f"{msg}\nCite only numbers from the available passages."
                )

        return ValidationResult(is_valid=True, data=output, raw_text=text)

    @classmethod
    def validate_query_rewrite(cls, raw_text: str) -> ValidationResult:
        """Validates Prompt 09 (09_query_rewriter.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the required keys.",
            )

        try:
            output = QueryRewriteOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nReturn valid JSON with at most 3 sub_queries.",
            )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_topic_interview(cls, raw_text: str) -> ValidationResult:
        """Validates Prompt 10 (10_topic_interviewer.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only.",
            )

        try:
            output = TopicInterviewOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nAsk ONE short plain-language question for the most important missing field, or set ready: true if none are missing.",
            )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_topic_recommendation(
        cls,
        raw_text: str,
        allowed_paper_ids: Optional[List[str]] = None,
        allowed_datasets: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validates Prompt 11 (11_topic_recommender.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the schema.",
            )

        try:
            output = TopicRecommendationOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nFix schema errors and never use 'novel', 'first', 'unexplored', or 'state of the art'.",
            )

        if allowed_paper_ids:
            allowed_set = {p.strip() for p in allowed_paper_ids if p.strip()}
            for opt in output.options:
                bad_ev = [p for p in opt.evidence_ids if p not in allowed_set]
                bad_read = [p for p in opt.papers_to_read if p not in allowed_set]
                if bad_ev or bad_read:
                    msg = f"Rule violation: Do not invent paper IDs. Unknown IDs: {bad_ev + bad_read}"
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nUse only paper IDs from the provided input.",
                    )

        if allowed_datasets:
            allowed_ds_low = {d.strip().lower() for d in allowed_datasets if d.strip()}
            for opt in output.options:
                bad_ds = [d for d in opt.datasets if d.strip().lower() not in allowed_ds_low]
                if bad_ds:
                    msg = f"Rule violation: Do not invent datasets. Unknown datasets: {bad_ds}"
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nUse only datasets from the provided input.",
                    )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_conflict_explainer(
        cls,
        raw_text: str,
        record_a_quote_source: str = "",
        record_b_quote_source: str = "",
    ) -> ValidationResult:
        """Validates Prompt 12 (12_conflict_explainer.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the schema.",
            )

        try:
            output = ConflictExplainerOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nKeep each quote <= 25 words with its page number.",
            )

        if record_a_quote_source and output.quote_a != "N/A":
            if output.quote_a.lower() not in record_a_quote_source.lower():
                msg = "Rule violation: quote_a must be a verbatim substring of Record A's text."
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nCopy quote_a verbatim from Record A (<= 25 words).",
                )

        if record_b_quote_source and output.quote_b != "N/A":
            if output.quote_b.lower() not in record_b_quote_source.lower():
                msg = "Rule violation: quote_b must be a verbatim substring of Record B's text."
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nCopy quote_b verbatim from Record B (<= 25 words).",
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_prior_work_evidence(
        cls,
        raw_text: str,
        allowed_result_ids: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validates Prompt 13 (13_prior_work_evidence.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the schema.",
            )

        try:
            output = PriorWorkEvidenceOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nList up to 5 closest papers and never judge novelty.",
            )

        if allowed_result_ids:
            allowed_set = {r.strip() for r in allowed_result_ids if r.strip()}
            bad_ids = [item.id for item in output.closest if item.id not in allowed_set]
            if bad_ids:
                msg = f"Rule violation: Do not invent result IDs. Unknown IDs: {bad_ids}"
                return ValidationResult(
                    is_valid=False,
                    data=output,
                    raw_json=parsed_json,
                    error_message=msg,
                    retry_prompt=f"{msg}\nUse only IDs from the provided search results.",
                )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_spoken_explanation(cls, raw_text: str) -> ValidationResult:
        """Validates Prompt 14 (14_spoken_explainer.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the schema.",
            )

        try:
            output = SpokenExplanationOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nKeep spoken_text <= 120 words, say 'candidate' instead of 'finding', and avoid symbols.",
            )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)

    @classmethod
    def validate_grounded_qa(
        cls,
        raw_text: str,
        passages_source: str = "",
    ) -> ValidationResult:
        """Validates Prompt 15 (15_grounded_qa.md) JSON output."""
        parsed_json, error = cls.parse_json_object(raw_text)
        if error or parsed_json is None:
            return ValidationResult(
                is_valid=False,
                error_message=error,
                retry_prompt=f"Error in previous response: {error}. Output JSON only matching the schema.",
            )

        try:
            output = GroundedQAOutput.model_validate(parsed_json)
        except ValidationError as ve:
            error_details = [f"Field '{' -> '.join(str(l) for l in err['loc'])}': {err['msg']}" for err in ve.errors()]
            msg = "; ".join(error_details)
            return ValidationResult(
                is_valid=False,
                raw_json=parsed_json,
                error_message=msg,
                retry_prompt=f"Validation errors:\n{msg}\nCite [paper_id, page] with <= 25-word verbatim quotes, or set answerable: false.",
            )

        if passages_source and output.answerable:
            src_low = passages_source.lower()
            for claim in output.claims:
                if claim.quote.lower() not in src_low:
                    msg = f"Rule violation: Quote must be copied verbatim from passages: '{claim.quote}'"
                    return ValidationResult(
                        is_valid=False,
                        data=output,
                        raw_json=parsed_json,
                        error_message=msg,
                        retry_prompt=f"{msg}\nCopy verbatim quotes (<= 25 words) directly from the provided passages.",
                    )

        return ValidationResult(is_valid=True, data=output, raw_json=parsed_json)
