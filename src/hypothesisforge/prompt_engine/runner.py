"""
HypothesisForge - Prompt Runner
Executes prompts against a language model, validates responses, manages 1x retry on invalid reply,
and executes deterministic rule-based fallback if retry fails.
Supports all Prompts 01 - 08.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
from pydantic import BaseModel

from .loader import PromptLoader, PromptDefinition
from .validator import ResponseValidator, ValidationResult
from .fallbacks import (
    RuleBasedLiteratureExtractor,
    RuleBasedGapEvaluator,
    RuleBasedContradictionEvaluator,
    RuleBasedHypothesisGenerator,
    RuleBasedScientificCritic,
    RuleBasedExperimentDesigner,
    RuleBasedReportSummarizer,
    RuleBasedEvidenceAnswerer,
    RuleBasedQueryRewriter,
    RuleBasedTopicInterviewer,
    RuleBasedTopicRecommender,
    RuleBasedConflictExplainer,
    RuleBasedPriorWorkReporter,
)
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
)


class PromptExecutionResult:
    """
    Standard envelope capturing execution details, prompt version used,
    validation status, retry count, and fallback status.
    """

    def __init__(
        self,
        prompt_name: str,
        prompt_version: str,
        data: Union[BaseModel, str],
        raw_output: str,
        retries_used: int = 0,
        used_fallback: bool = False,
        prompt_used: bool = True,
        validation_error: Optional[str] = None
    ):
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.data = data
        self.raw_output = raw_output
        self.retries_used = retries_used
        self.used_fallback = used_fallback
        self.prompt_used = prompt_used
        self.validation_error = validation_error

    def to_dict(self) -> Dict[str, Any]:
        data_serialized = self.data.model_dump() if isinstance(self.data, BaseModel) else self.data
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version if self.prompt_used else "not used",
            "prompt_used": self.prompt_used,
            "used_fallback": self.used_fallback,
            "retries_used": self.retries_used,
            "validation_error": self.validation_error,
            "data": data_serialized,
        }


LLMCallable = Callable[[str, str], str]  # (system_prompt, user_prompt) -> response_str


class PromptRunner:
    """
    Manages end-to-end execution of Prompts 01 through 08.
    """

    def __init__(self, loader: Optional[PromptLoader] = None):
        self.loader = loader or PromptLoader()

    def run_literature_extraction(
        self,
        title: str,
        year: Union[int, str],
        paper_text: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 01: 01_literature_extraction.md"""
        prompt_def = self.loader.load_prompt("01_literature_extraction")
        user_prompt = prompt_def.render_user_prompt(title=title, year=str(year), paper_text=paper_text)

        if llm_fn is None:
            fallback_data = RuleBasedLiteratureExtractor.extract(title, year, paper_text)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based extraction executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_literature_extraction(raw_reply, source_text=paper_text)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_literature_extraction(retry_reply, source_text=paper_text)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedLiteratureExtractor.extract(title, year, paper_text)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_gap_relevance(
        self,
        gap: str,
        related_records: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 02: 02_gap_relevance.md"""
        prompt_def = self.loader.load_prompt("02_gap_relevance")
        user_prompt = prompt_def.render_user_prompt(gap=gap, related_records=related_records)

        if llm_fn is None:
            fallback_data = RuleBasedGapEvaluator.evaluate(gap, related_records)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based gap evaluation executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_gap_relevance(raw_reply)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_gap_relevance(retry_reply)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedGapEvaluator.evaluate(gap, related_records)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_contradiction_judgement(
        self,
        pair: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 03: 03_contradiction_judgement.md"""
        prompt_def = self.loader.load_prompt("03_contradiction_judgement")
        user_prompt = prompt_def.render_user_prompt(pair=pair)

        if llm_fn is None:
            fallback_data = RuleBasedContradictionEvaluator.evaluate(pair)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based contradiction evaluation executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_contradiction_judgement(raw_reply, pair_context=pair)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_contradiction_judgement(retry_reply, pair_context=pair)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedContradictionEvaluator.evaluate(pair)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_hypothesis_generation(
        self,
        gap: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 04: 04_hypothesis_generation.md"""
        prompt_def = self.loader.load_prompt("04_hypothesis_generation")
        user_prompt = prompt_def.render_user_prompt(gap=gap)

        if llm_fn is None:
            fallback_data = RuleBasedHypothesisGenerator.generate(gap)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based hypothesis generation executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_hypothesis_generation(raw_reply, evidence_context=gap)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_hypothesis_generation(retry_reply, evidence_context=gap)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedHypothesisGenerator.generate(gap)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_scientific_critic(
        self,
        hypothesis: str,
        checks: str,
        prior_status: str,
        overlap: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 05: 05_scientific_critic.md"""
        prompt_def = self.loader.load_prompt("05_scientific_critic")
        user_prompt = prompt_def.render_user_prompt(
            hypothesis=hypothesis, checks=checks, prior_status=prior_status, overlap=overlap
        )

        if llm_fn is None:
            fallback_data = RuleBasedScientificCritic.evaluate(hypothesis, checks, prior_status, overlap)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based scientific critic executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_scientific_critic(
            raw_reply, prior_status=prior_status, overlap=overlap, checks=checks
        )

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_scientific_critic(
            retry_reply, prior_status=prior_status, overlap=overlap, checks=checks
        )

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedScientificCritic.evaluate(hypothesis, checks, prior_status, overlap)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_experiment_design(
        self,
        hypothesis: str,
        draft_plan: str,
        grounded_options: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 06: 06_experiment_design.md"""
        prompt_def = self.loader.load_prompt("06_experiment_design")
        user_prompt = prompt_def.render_user_prompt(
            hypothesis=hypothesis, draft_plan=draft_plan, grounded_options=grounded_options
        )

        if llm_fn is None:
            fallback_data = RuleBasedExperimentDesigner.design(hypothesis, draft_plan, grounded_options)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based experiment design executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_experiment_design(raw_reply, grounded_options=grounded_options)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_experiment_design(retry_reply, grounded_options=grounded_options)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedExperimentDesigner.design(hypothesis, draft_plan, grounded_options)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_report_summary(
        self,
        facts: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 07: 07_report_summary.md"""
        prompt_def = self.loader.load_prompt("07_report_summary")
        user_prompt = prompt_def.render_user_prompt(facts=facts)

        if llm_fn is None:
            fallback_data = RuleBasedReportSummarizer.summarize(facts)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output=fallback_data.text,
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_report_summary(raw_reply, facts_json=facts)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_report_summary(retry_reply, facts_json=facts)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedReportSummarizer.summarize(facts)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_evidence_answer(
        self,
        question: str,
        passages: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 08: 08_evidence_answer.md"""
        prompt_def = self.loader.load_prompt("08_evidence_answer")
        user_prompt = prompt_def.render_user_prompt(question=question, passages=passages)

        if llm_fn is None:
            fallback_data = RuleBasedEvidenceAnswerer.answer(question, passages)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output=fallback_data.text,
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_evidence_answer(raw_reply, passages=passages)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_evidence_answer(retry_reply, passages=passages)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False
            )

        fallback_data = RuleBasedEvidenceAnswerer.answer(question, passages)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message
        )

    def run_query_rewrite(
        self,
        history: str,
        question: str,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 09: 09_query_rewriter.md"""
        prompt_def = self.loader.load_prompt("09_query_rewriter")
        user_prompt = prompt_def.render_user_prompt(history=history or "None", question=question)

        if llm_fn is None:
            fallback_data = RuleBasedQueryRewriter.rewrite(history, question)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based query rewriter executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_query_rewrite(raw_reply)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False,
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_query_rewrite(retry_reply)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False,
            )

        fallback_data = RuleBasedQueryRewriter.rewrite(history, question)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message,
        )

    def run_topic_interview(
        self,
        answers: Any,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 10: 10_topic_interviewer.md"""
        import json as _json

        prompt_def = self.loader.load_prompt("10_topic_interviewer")
        answers_str = _json.dumps(answers, ensure_ascii=False) if isinstance(answers, dict) else str(answers or "{}")
        user_prompt = prompt_def.render_user_prompt(answers=answers_str)

        if llm_fn is None:
            fallback_data = RuleBasedTopicInterviewer.interview(answers)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based topic interviewer executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_topic_interview(raw_reply)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False,
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_topic_interview(retry_reply)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False,
            )

        fallback_data = RuleBasedTopicInterviewer.interview(answers)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message,
        )

    def run_topic_recommendation(
        self,
        profile: Any,
        gaps: Any,
        contradictions: Any,
        scope_criteria: Any,
        n_papers: int,
        allowed_paper_ids: Optional[list] = None,
        allowed_datasets: Optional[list] = None,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 11: 11_topic_recommender.md"""
        import json as _json

        prompt_def = self.loader.load_prompt("11_topic_recommender")
        profile_str = _json.dumps(profile, ensure_ascii=False) if isinstance(profile, dict) else str(profile)
        gaps_str = _json.dumps(gaps, ensure_ascii=False) if isinstance(gaps, (list, dict)) else str(gaps)
        con_str = _json.dumps(contradictions, ensure_ascii=False) if isinstance(contradictions, (list, dict)) else str(contradictions)
        scope_str = _json.dumps(scope_criteria, ensure_ascii=False) if isinstance(scope_criteria, dict) else str(scope_criteria)

        user_prompt = prompt_def.render_user_prompt(
            profile=profile_str,
            gaps=gaps_str,
            contradictions=con_str,
            scope_criteria=scope_str,
            n=str(n_papers),
        )

        if llm_fn is None:
            fallback_data = RuleBasedTopicRecommender.recommend(
                gaps=gaps,
                contradictions=contradictions,
                profile=profile,
                scope_criteria=scope_criteria,
                n_papers=n_papers,
            )
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based topic recommender executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_topic_recommendation(
            raw_reply,
            allowed_paper_ids=allowed_paper_ids,
            allowed_datasets=allowed_datasets,
        )

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False,
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_topic_recommendation(
            retry_reply,
            allowed_paper_ids=allowed_paper_ids,
            allowed_datasets=allowed_datasets,
        )

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False,
            )

        fallback_data = RuleBasedTopicRecommender.recommend(
            gaps=gaps,
            contradictions=contradictions,
            profile=profile,
            scope_criteria=scope_criteria,
            n_papers=n_papers,
        )
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message,
        )

    def run_conflict_explainer(
        self,
        record_a: Any,
        record_b: Any,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 12: 12_conflict_explainer.md"""
        import json as _json

        prompt_def = self.loader.load_prompt("12_conflict_explainer")
        rec_a_str = _json.dumps(record_a, ensure_ascii=False) if isinstance(record_a, dict) else str(record_a)
        rec_b_str = _json.dumps(record_b, ensure_ascii=False) if isinstance(record_b, dict) else str(record_b)
        user_prompt = prompt_def.render_user_prompt(record_a=rec_a_str, record_b=rec_b_str)

        if llm_fn is None:
            fallback_data = RuleBasedConflictExplainer.explain(record_a, record_b)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based conflict explainer executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_conflict_explainer(
            raw_reply,
            record_a_quote_source=rec_a_str,
            record_b_quote_source=rec_b_str,
        )

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False,
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_conflict_explainer(
            retry_reply,
            record_a_quote_source=rec_a_str,
            record_b_quote_source=rec_b_str,
        )

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False,
            )

        fallback_data = RuleBasedConflictExplainer.explain(record_a, record_b)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message,
        )

    def run_prior_work_evidence(
        self,
        hypothesis: str,
        results: Any,
        allowed_result_ids: Optional[list] = None,
        llm_fn: Optional[LLMCallable] = None,
    ) -> PromptExecutionResult:
        """Runs Prompt 13: 13_prior_work_evidence.md"""
        import json as _json

        prompt_def = self.loader.load_prompt("13_prior_work_evidence")
        results_list = results if isinstance(results, list) else []
        n_results = len(results_list)
        results_str = _json.dumps(results, ensure_ascii=False) if isinstance(results, (list, dict)) else str(results)
        user_prompt = prompt_def.render_user_prompt(
            hypothesis=hypothesis,
            results=results_str,
            n=str(n_results),
        )

        if llm_fn is None:
            fallback_data = RuleBasedPriorWorkReporter.report(hypothesis, results)
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version="not used",
                data=fallback_data,
                raw_output="[No LLM configured - Rule-based prior-work reporter executed]",
                retries_used=0,
                used_fallback=True,
                prompt_used=False,
            )

        raw_reply = llm_fn(prompt_def.system_prompt, user_prompt)
        val = ResponseValidator.validate_prior_work_evidence(raw_reply, allowed_result_ids=allowed_result_ids)

        if val.is_valid and val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=val.data,
                raw_output=raw_reply,
                retries_used=0,
                used_fallback=False,
            )

        retry_prompt = f"{user_prompt}\n\nIMPORTANT CORRECTION: {val.retry_prompt}"
        retry_reply = llm_fn(prompt_def.system_prompt, retry_prompt)
        retry_val = ResponseValidator.validate_prior_work_evidence(retry_reply, allowed_result_ids=allowed_result_ids)

        if retry_val.is_valid and retry_val.data:
            return PromptExecutionResult(
                prompt_name=prompt_def.name,
                prompt_version=prompt_def.version,
                data=retry_val.data,
                raw_output=retry_reply,
                retries_used=1,
                used_fallback=False,
            )

        fallback_data = RuleBasedPriorWorkReporter.report(hypothesis, results)
        return PromptExecutionResult(
            prompt_name=prompt_def.name,
            prompt_version=prompt_def.version,
            data=fallback_data,
            raw_output=retry_reply,
            retries_used=1,
            used_fallback=True,
            validation_error=retry_val.error_message or val.error_message,
        )
