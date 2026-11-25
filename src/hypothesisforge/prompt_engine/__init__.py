"""
HypothesisForge - Prompt Engine Package (Prompts 01 - 08 Complete)
"""

from .models import (
    ReportedResultItem,
    FindingItem,
    LimitationItem,
    LiteratureExtractionOutput,
    GapRelevanceOutput,
    ContradictionJudgementOutput,
    HypothesisGenerationOutput,
    ScientificCriticOutput,
    ExperimentDesignOutput,
    ReportSummaryOutput,
    EvidenceAnswerOutput,
)
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
)
from .runner import PromptRunner, PromptExecutionResult

__all__ = [
    "ReportedResultItem",
    "FindingItem",
    "LimitationItem",
    "LiteratureExtractionOutput",
    "GapRelevanceOutput",
    "ContradictionJudgementOutput",
    "HypothesisGenerationOutput",
    "ScientificCriticOutput",
    "ExperimentDesignOutput",
    "ReportSummaryOutput",
    "EvidenceAnswerOutput",
    "PromptLoader",
    "PromptDefinition",
    "ResponseValidator",
    "ValidationResult",
    "RuleBasedLiteratureExtractor",
    "RuleBasedGapEvaluator",
    "RuleBasedContradictionEvaluator",
    "RuleBasedHypothesisGenerator",
    "RuleBasedScientificCritic",
    "RuleBasedExperimentDesigner",
    "RuleBasedReportSummarizer",
    "RuleBasedEvidenceAnswerer",
    "PromptRunner",
    "PromptExecutionResult",
]
