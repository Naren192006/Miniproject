"""
HypothesisForge - Prompt Engine Models
Implements strict Pydantic models for validating LLM responses against the Prompt Pack specifications (Prompts 01 - 08).
"""

import re
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Helper functions & Constants
# ==============================================================================

def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.strip().split()) if text else 0


FORBIDDEN_NOVELTY_PHRASES = [
    "is novel",
    "are novel",
    "is globally novel",
    "claim of novelty",
    "entirely novel",
    "nobody has tested",
    "no one has tested",
    "never been tested",
    "completely new discovery",
    "groundbreaking discovery",
    "unprecedented discovery",
    "is completely novel",
    "truly novel",
    "a discovery",
    "new discovery",
    "unexplored gap",
    "is unexplored",
]

FORBIDDEN_PAPER_WRONG_PHRASES = [
    "is wrong",
    "are wrong",
    "was wrong",
    "paper a is wrong",
    "paper b is wrong",
    "paper 1 is wrong",
    "paper 2 is wrong",
    "is incorrect",
    "is flawed",
    "paper is mistaken",
    "fabricated results",
    "false results",
]

FORBIDDEN_FACT_PHRASES = [
    "will definitely prove",
    "is a proven fact that",
    "proves as fact",
    "proven beyond doubt",
    "will certainly show",
    "indisputably demonstrates",
]


# ==============================================================================
# Prompt 01: Literature Extraction Models
# ==============================================================================

class ReportedResultItem(BaseModel):
    value: Union[float, int, str] = Field(..., description="The number as written, without the % sign")
    metric: str = Field(..., min_length=1, description="The metric this number measures")
    dataset: str = Field(..., min_length=1, description="The dataset it was measured on")
    setting: str = Field(..., min_length=1, description="Architecture, training strategy and shift setting phrase")
    method: Literal["proposed", "baseline"] = Field(..., description="'proposed' or 'baseline'")
    page: int = Field(..., ge=1, description="Page number as an integer")
    quote: str = Field(..., min_length=1, description="Verbatim sentence or table row, at most 25 words")

    @field_validator("quote")
    @classmethod
    def validate_quote_word_count(cls, v: str) -> str:
        words = count_words(v)
        if words > 25:
            raise ValueError(f"Reported result quote exceeds 25 words ({words} words): '{v[:50]}...'")
        return v

    @field_validator("value")
    @classmethod
    def strip_percent(cls, v: Union[float, int, str]) -> Union[float, int, str]:
        if isinstance(v, str):
            v_clean = v.replace("%", "").strip()
            try:
                if "." in v_clean:
                    return float(v_clean)
                return int(v_clean)
            except ValueError:
                return v_clean
        return v


class FindingItem(BaseModel):
    text: str = Field(..., min_length=1, description="One short sentence paraphrasing a reported result")
    page: int = Field(..., ge=1, description="Page number")
    section: str = Field(..., description="Section heading or 'not stated'")
    quote: str = Field(..., min_length=1, description="Verbatim passage of at most 25 words")

    @field_validator("quote")
    @classmethod
    def validate_quote_word_count(cls, v: str) -> str:
        words = count_words(v)
        if words > 25:
            raise ValueError(f"Finding quote exceeds 25 words ({words} words): '{v[:50]}...'")
        return v


class LimitationItem(BaseModel):
    text: str = Field(..., min_length=1, description="Limitation text")
    page: int = Field(..., ge=1, description="Page number")
    section: str = Field(..., description="Section heading or 'not stated'")
    quote: str = Field(..., min_length=1, description="Verbatim quote, at most 25 words")
    origin: Literal["author_stated", "inferred"] = Field(..., description="'author_stated' or 'inferred'")

    @field_validator("quote")
    @classmethod
    def validate_quote_word_count(cls, v: str) -> str:
        words = count_words(v)
        if words > 25:
            raise ValueError(f"Limitation quote exceeds 25 words ({words} words): '{v[:50]}...'")
        return v


class LiteratureExtractionOutput(BaseModel):
    """Schema for Prompt 01: 01_literature_extraction.md"""
    tasks: List[str] = Field(default_factory=list)
    architectures: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    training_strategies: List[str] = Field(default_factory=list)
    shift_settings: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    evaluation_protocol: str = Field(...)
    reported_results: List[ReportedResultItem] = Field(default_factory=list, max_length=8)
    findings: List[FindingItem] = Field(default_factory=list, max_length=6)
    limitations: List[LimitationItem] = Field(default_factory=list, max_length=6)
    extraction_confidence: Literal["high", "medium", "low"] = Field(...)


# ==============================================================================
# Prompt 02: Gap Relevance Models
# ==============================================================================

class GapRelevanceOutput(BaseModel):
    """Schema for Prompt 02: 02_gap_relevance.md"""
    rating: Literal["high", "medium", "low", "not_meaningful"] = Field(...)
    reason: str = Field(..., min_length=10)
    absence_explanation: Literal[
        "corpus size",
        "impossible or meaningless pairing",
        "not reported by these papers but plausible",
        "unknown"
    ] = Field(...)
    questions_to_check: List[str] = Field(default_factory=list, max_length=3)

    @field_validator("reason")
    @classmethod
    def validate_no_novelty_claims(cls, v: str) -> str:
        lower = v.lower()
        for phrase in FORBIDDEN_NOVELTY_PHRASES:
            if phrase in lower:
                raise ValueError(
                    f"Rule violation: No prompt may claim that a finding is novel or that nobody has tested it. Found: '{phrase}'"
                )
        return v

    @model_validator(mode="after")
    def validate_thin_evidence_rule(self) -> "GapRelevanceOutput":
        reason_lower = self.reason.lower()
        if "evidence is thin" in reason_lower or "evidence is too thin" in reason_lower or "thin evidence" in reason_lower:
            if self.rating != "low":
                raise ValueError(
                    f"Rule violation: If the evidence is too thin to judge, rating must be 'low' (got '{self.rating}')"
                )
        return self


# ==============================================================================
# Prompt 03: Contradiction Judgement Models
# ==============================================================================

class ContradictionJudgementOutput(BaseModel):
    """Schema for Prompt 03: 03_contradiction_judgement.md"""
    classification: Literal[
        "apparent agreement",
        "apparent disagreement under comparable settings",
        "non-comparable evidence",
        "insufficient information"
    ] = Field(...)
    explanation: str = Field(..., min_length=15)
    conditions_to_verify: List[str] = Field(default_factory=list, max_length=4)
    confidence_score: float = Field(default=0.75, ge=0.0, le=1.0)
    conflict_factors: List[str] = Field(default_factory=list)

    @field_validator("explanation")
    @classmethod
    def validate_never_declare_wrong(cls, v: str) -> str:
        lower = v.lower()
        for phrase in FORBIDDEN_PAPER_WRONG_PHRASES:
            if phrase in lower:
                raise ValueError(f"Rule violation: Never declare either paper wrong. Found: '{phrase}'")
        return v


# ==============================================================================
# Prompt 04: Hypothesis Generation Models
# ==============================================================================

class HypothesisGenerationOutput(BaseModel):
    """Schema for Prompt 04: 04_hypothesis_generation.md"""
    decision: Literal["hypothesis", "insufficient_evidence"] = Field(...)
    statement: Optional[str] = Field(default=None)
    expected_direction: Optional[Literal["increase", "decrease", "no change", "unknown"]] = Field(default=None)
    baseline: Optional[str] = Field(default=None)
    metric: Optional[str] = Field(default=None)
    assumptions: Optional[List[str]] = Field(default=None)
    cited_paper_ids: Optional[List[str]] = Field(default=None)
    reason: str = Field(..., min_length=10)

    @model_validator(mode="after")
    def validate_decision_fields(self) -> "HypothesisGenerationOutput":
        full_text = f"{self.statement or ''} {self.reason}".lower()
        for phrase in FORBIDDEN_NOVELTY_PHRASES:
            if phrase in full_text:
                raise ValueError(f"Rule violation: Do not claim the hypothesis is novel. Found: '{phrase}'")

        if self.decision == "hypothesis":
            if not self.statement:
                raise ValueError("Field 'statement' is required when decision is 'hypothesis'")
            stmt_lower = self.statement.lower()
            if not (stmt_lower.startswith("if ") and "then " in stmt_lower and "measured by " in stmt_lower and "relative to " in stmt_lower):
                raise ValueError("Statement must be in the form 'If ..., then ..., measured by ..., relative to ...'")

            if not self.expected_direction:
                raise ValueError("Field 'expected_direction' is required when decision is 'hypothesis'")
            if not self.baseline:
                raise ValueError("Field 'baseline' is required when decision is 'hypothesis'")
            if not self.metric:
                raise ValueError("Field 'metric' is required when decision is 'hypothesis'")

            if "improve" in stmt_lower:
                if self.metric.lower() not in stmt_lower:
                    raise ValueError(
                        f"Rule violation: Avoid vague verbs such as 'improve' unless the metric ('{self.metric}') is named in the statement."
                    )

            if self.assumptions is not None and len(self.assumptions) > 4:
                raise ValueError(f"Assumptions list exceeds maximum of 4 items ({len(self.assumptions)} given)")

            if not self.cited_paper_ids or len(self.cited_paper_ids) == 0:
                raise ValueError("Field 'cited_paper_ids' must not be empty when decision is 'hypothesis'")

        elif self.decision == "insufficient_evidence":
            if not self.reason:
                raise ValueError("Field 'reason' is required when decision is 'insufficient_evidence'")

        return self


# ==============================================================================
# Prompt 05: Scientific Critic Models
# ==============================================================================

class ScientificCriticOutput(BaseModel):
    """Schema for Prompt 05: 05_scientific_critic.md"""
    verdict: Literal["accept", "revise", "reject"] = Field(...)
    rationale: str = Field(..., min_length=15)
    missing_evidence: List[str] = Field(default_factory=list, max_length=4)
    alternative_explanations: List[str] = Field(default_factory=list, max_length=4)

    @field_validator("rationale")
    @classmethod
    def validate_no_novelty_claim(cls, v: str) -> str:
        lower = v.lower()
        for phrase in FORBIDDEN_NOVELTY_PHRASES:
            if phrase in lower:
                raise ValueError(
                    f"Rule violation: Even when accepted, claim is not verified as globally novel. Found: '{phrase}'"
                )
        return v


# ==============================================================================
# Prompt 06: Experiment Design Models
# ==============================================================================

class ExperimentDesignOutput(BaseModel):
    """Schema for Prompt 06: 06_experiment_design.md"""
    research_question: str = Field(..., min_length=10)
    independent_variable: str = Field(..., min_length=3)
    dependent_variable: str = Field(..., min_length=3)
    dataset_and_split: str = Field(..., min_length=5)
    baselines: List[str] = Field(..., min_length=1)
    controls: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(..., min_length=1)
    procedure: List[str] = Field(..., min_length=1, max_length=7)
    seeds_and_runs: str = Field(..., min_length=10)
    statistics: str = Field(..., min_length=10)
    compute_estimate: str = Field(..., min_length=5)
    threats_to_validity: List[str] = Field(default_factory=list)
    expected_observations: str = Field(..., min_length=10)
    limitations: List[str] = Field(default_factory=list)

    @field_validator("expected_observations")
    @classmethod
    def validate_no_facts_in_expectations(cls, v: str) -> str:
        lower = v.lower()
        for phrase in FORBIDDEN_FACT_PHRASES:
            if phrase in lower:
                raise ValueError(f"Rule violation: Do not state an expected result as fact. Found: '{phrase}'")
        return v

    @field_validator("seeds_and_runs")
    @classmethod
    def validate_minimum_seeds(cls, v: str) -> str:
        lower = v.lower()
        seed_numbers = re.findall(r"\b(\d+)\s*(?:random\s*)?seeds?\b", lower)
        if seed_numbers:
            counts = [int(n) for n in seed_numbers]
            if max(counts) < 3:
                raise ValueError(
                    f"Rule violation: Must state at least three random seeds (found {max(counts)} in '{v}')"
                )
        elif not any(w in lower for w in ["three", "four", "five", "3", "4", "5", "10"]):
            raise ValueError(f"Rule violation: seeds_and_runs must state at least three random seeds: '{v}'")
        return v


# ==============================================================================
# Prompt 07: Report Summary Models (Plain Text Wrapper)
# ==============================================================================

class ReportSummaryOutput(BaseModel):
    """
    Validation envelope for Prompt 07: 07_report_summary.md
    Plain-text response of no more than 180 words in three short paragraphs.
    """
    text: str = Field(..., min_length=20, description="Full plain-text summary")
    paragraphs: List[str] = Field(..., description="The three extracted paragraphs")
    word_count: int = Field(..., le=180, description="Word count (max 180 words)")

    @field_validator("text")
    @classmethod
    def validate_no_bullets_or_headings(cls, v: str) -> str:
        # Check for headings (#, ##)
        if re.search(r"(?:^|\n)\s*#{1,6}\s+", v):
            raise ValueError("Rule violation: No headings allowed in report summary.")
        if re.search(r"(?:^|\n)\s*paragraph\s+\d+[:\.]?", v, re.I):
            raise ValueError("Rule violation: No headings or paragraph labels allowed in report summary.")

        # Check for bullet points (-, *, •, 1.)
        if re.search(r"(?:^|\n)\s*[-*•]\s+", v) or re.search(r"(?:^|\n)\s*\d+\.\s+", v):
            raise ValueError("Rule violation: No bullet points or numbered lists allowed in report summary.")

        # Check forbidden novelty claims
        lower = v.lower()
        if "a discovery" in lower or "new discovery" in lower or "unexplored" in lower:
            raise ValueError("Rule violation: Do not say that a gap is new, unexplored or a discovery.")

        return v

    @field_validator("paragraphs")
    @classmethod
    def validate_three_paragraphs(cls, v: List[str]) -> List[str]:
        if len(v) != 3:
            raise ValueError(f"Rule violation: Summary must be exactly three paragraphs (got {len(v)} paragraphs).")
        return v


# ==============================================================================
# Prompt 08: Evidence Answer Models (Plain Text Wrapper)
# ==============================================================================

class EvidenceAnswerOutput(BaseModel):
    """
    Validation envelope for Prompt 08: 08_evidence_answer.md
    Plain-text response in one to four short paragraphs citing passages in square brackets.
    """
    text: str = Field(..., min_length=5, description="Full plain-text answer")
    paragraphs: List[str] = Field(..., description="Paragraphs (1 to 4)")
    citations: List[int] = Field(default_factory=list, description="Extracted citation numbers e.g. [2] -> 2")
    is_declined: bool = Field(default=False, description="True if passages do not answer the question")

    @model_validator(mode="after")
    def validate_evidence_answer_rules(self) -> "EvidenceAnswerOutput":
        clean_text = self.text.strip()
        decline_phrase = "the passages do not answer this question"

        if decline_phrase in clean_text.lower():
            self.is_declined = True
            return self

        # If answered: 1 to 4 paragraphs
        if not (1 <= len(self.paragraphs) <= 4):
            raise ValueError(f"Rule violation: Answer must be in one to four short paragraphs (got {len(self.paragraphs)}).")

        # Must have at least one passage citation [n]
        if not self.citations:
            raise ValueError("Rule violation: Cite every claim with a passage number in square brackets (e.g. [1]).")

        return self


# ==============================================================================
# Prompt 09: Query Rewriter & Intent Router Models
# ==============================================================================

class QueryRewriteOutput(BaseModel):
    """Schema for Prompt 09: 09_query_rewriter.md"""
    standalone_question: str = Field(..., min_length=3, description="Self-contained rewritten question")
    intent: Literal[
        "summary",
        "compare",
        "gap",
        "contradiction",
        "topic_advice",
        "factual_qa",
        "out_of_scope",
    ] = Field(..., description="Classified user intent")
    sub_queries: List[str] = Field(..., min_length=1, max_length=3, description="At most 3 sub-queries")
    needs_external_search: bool = Field(default=False, description="True only if corpus unlikely to contain answer")
