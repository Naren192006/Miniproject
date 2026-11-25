"""
Tests for HypothesisForge Ingestion, Pipeline Orchestration, Experimental Matrix, and Knowledge Graph.
"""

from pathlib import Path
from hypothesisforge.ingestion import retrieve_passages
from hypothesisforge.llm_client import get_llm_callable_from_env
from hypothesisforge.pipeline import get_sample_corpus, run_full_pipeline


def test_env_llm_config_defaults(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=rule_based\n", encoding="utf-8")
    label, fn = get_llm_callable_from_env(env_path=env_file)
    assert label == "Rule-Based Analysis"
    assert fn is None

    env_file.write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=test-key\nGEMINI_MODEL=gemini-2.5-flash\n", encoding="utf-8")
    label2, fn2 = get_llm_callable_from_env(env_path=env_file)
    assert label2 == "Gemini (gemini-2.5-flash)"
    assert fn2 is not None


def test_sample_corpus_and_full_pipeline_offline():
    papers = get_sample_corpus()
    assert len(papers) == 3

    result = run_full_pipeline(papers, llm_fn=None)
    assert len(result.literature_records) == 3
    assert len(result.candidate_gaps) > 0
    assert len(result.gap_assessments) > 0
    assert len(result.contradictions) > 0
    assert len(result.hypotheses) > 0
    assert len(result.critic_reviews) > 0
    assert len(result.experiment_plans) > 0
    assert result.report_summary is not None
    assert result.report_summary.word_count <= 180
    assert len(result.knowledge_graph.nodes) > 0
    assert len(result.audit_log) > 0


def test_passage_retrieval():
    papers = get_sample_corpus()
    all_chunks = [c for p in papers for c in p.chunks]
    passages_str, selected = retrieve_passages("What accuracy does ViT-B/16 achieve on PACS?", all_chunks, top_k=3)

    assert len(selected) > 0
    assert "[1]" in passages_str
    assert "86.2%" in passages_str
    assert "P-" in passages_str


def test_streamlit_app_syntax_valid():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    source = app_path.read_text(encoding="utf-8")
    compile(source, str(app_path), "exec")
