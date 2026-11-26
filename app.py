"""
HypothesisForge - Light-Mode Research Workbench (Streamlit Frontend)
Clean, flat, high-contrast light UI with zero dark themes and zero gradients.
"""

import html
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# Ensure src/ is importable when running `streamlit run app.py`
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hypothesisforge.experiment_runner import execute_experiment_plan, generate_experiment_script
from hypothesisforge.grounding_benchmark import run_full_evaluation_harness
from hypothesisforge.ingestion import PaperDocument, extract_pdf_bytes
from hypothesisforge.scope_advisor import ScopeInterview, recommend_scoped_topics
from hypothesisforge.llm_client import get_llm_callable_from_env
from hypothesisforge.paper_search import discover_papers, fetch_paper_documents
from hypothesisforge.prompt_engine.runner import PromptRunner
from hypothesisforge.pipeline import (
    AnalysisPipelineResult,
    answer_workspace_question,
    compute_corpus_signature,
    get_sample_corpus,
    run_full_pipeline,
)


# ==============================================================================
# Page Config & Light-Mode Flat CSS (Zero Dark Colours, Zero Gradients)
# ==============================================================================

st.set_page_config(
    page_title="HypothesisForge — Research Gap & Hypothesis Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_FLAT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0F172A;
    background-color: #FFFFFF;
}

/* Main container spacing */
.block-container {
    padding-top: 1.75rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

/* Sidebar flat light surface */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

/* Hide default Streamlit branding footer, menu & deploy button (keep sidebar toggle visible) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none !important; }
.stAppDeployButton { display: none !important; }

/* Prevent faded/dimmed content during Streamlit reruns */
[data-testid="stAppViewBlockContainer"],
.element-container,
.stMarkdown {
    opacity: 1 !important;
    transition: none !important;
}

/* Top header bar */
.hf-header {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.hf-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hf-subtitle {
    font-size: 0.92rem;
    color: #475569;
    margin: 0;
}

/* Flat KPI Metric Cards */
.hf-kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.75rem;
}
.hf-kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 0.35rem;
}
.hf-kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}
.hf-kpi-note {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* Flat Content Cards */
.hf-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 0.9rem;
}
.hf-card-muted {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}
.hf-card-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 0.45rem;
}
.hf-quote {
    background-color: #F8FAFC;
    border-left: 3px solid #2563EB;
    padding: 0.55rem 0.85rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #334155;
    border-radius: 0 6px 6px 0;
}
.hf-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* Flat Semantic Pills (Light Pastel + Crisp Text) */
.hf-pill {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid transparent;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
}
.hf-pill-green {
    background-color: #ECFDF5;
    color: #065F46;
    border-color: #A7F3D0;
}
.hf-pill-amber {
    background-color: #FFFBEB;
    color: #92400E;
    border-color: #FDE68A;
}
.hf-pill-red {
    background-color: #FEF2F2;
    color: #991B1B;
    border-color: #FECACA;
}
.hf-pill-blue {
    background-color: #EFF6FF;
    color: #1E40AF;
    border-color: #BFDBFE;
}
.hf-pill-slate {
    background-color: #F1F5F9;
    color: #334155;
    border-color: #CBD5E1;
}

/* Clean Tab Bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px 8px 0 0;
    color: #475569;
    font-weight: 500;
    padding: 0.45rem 0.95rem;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #2563EB !important;
    border-color: #CBD5E1 !important;
    border-bottom: 2px solid #2563EB !important;
    font-weight: 600;
}

/* Primary Buttons */
.stButton > button[kind="primary"] {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #1D4ED8;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8;
    border-color: #1E40AF;
}
</style>
"""
st.markdown(LIGHT_FLAT_CSS, unsafe_allow_html=True)


# ==============================================================================
# UI Helpers
# ==============================================================================

def esc(text: object) -> str:
    return html.escape(str(text))


def pill(label: str, variant: str = "slate") -> str:
    return f'<span class="hf-pill hf-pill-{variant}">{esc(label)}</span>'


def verdict_pill(value: str) -> str:
    v = value.lower()
    if v in {"accept", "high", "proposed", "author_stated", "apparent agreement"}:
        return pill(value, "green")
    if v in {"revise", "medium", "not reported by these papers but plausible", "inferred"}:
        return pill(value, "amber")
    if v in {"reject", "low", "not_meaningful", "insufficient_evidence"}:
        return pill(value, "red")
    return pill(value, "blue")


def render_knowledge_graph(
    G: nx.DiGraph,
    visible_types: List[str],
    show_edge_labels: bool = False,
) -> plt.Figure:
    """Renders a clean, light-mode NetworkX knowledge graph with filtering and non-overlapping layout."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    selected_nodes = [n for n in G.nodes if G.nodes[n].get("node_type", "Method") in visible_types]
    subG = G.subgraph(selected_nodes).copy()

    if len(subG.nodes) == 0:
        ax.text(0.5, 0.5, "No graph nodes match the selected filters.", ha="center", va="center", color="#64748B", fontsize=11)
        ax.axis("off")
        return fig

    pos = nx.spring_layout(subG, k=1.8, iterations=80, seed=42)

    color_map = {
        "Paper": "#DBEAFE",
        "Architecture": "#DCFCE7",
        "Dataset": "#FEF3C7",
        "Method": "#F1F5F9",
        "Hypothesis": "#FCE7F3",
    }
    edge_color_map = {
        "Paper": "#93C5FD",
        "Architecture": "#86EFAC",
        "Dataset": "#FCD34D",
        "Method": "#CBD5E1",
        "Hypothesis": "#F9A8D4",
    }

    node_colors = [color_map.get(subG.nodes[n].get("node_type", "Method"), "#F1F5F9") for n in subG.nodes]
    node_borders = [edge_color_map.get(subG.nodes[n].get("node_type", "Method"), "#CBD5E1") for n in subG.nodes]

    nx.draw_networkx_edges(
        subG,
        pos,
        ax=ax,
        edge_color="#CBD5E1",
        arrows=True,
        arrowsize=12,
        width=1.2,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(
        subG,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=1.6,
        node_size=1300,
    )
    nx.draw_networkx_labels(
        subG,
        pos,
        ax=ax,
        font_size=8,
        font_weight="bold",
        font_color="#0F172A",
    )

    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(subG, "relation")
        nx.draw_networkx_edge_labels(
            subG,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=6.5,
            font_color="#64748B",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="none", alpha=0.85),
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=ntype, markerfacecolor=col, markeredgecolor=edge_color_map[ntype], markersize=9)
        for ntype, col in color_map.items()
        if ntype in visible_types
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=True,
            facecolor="#F8FAFC",
            edgecolor="#E2E8F0",
            fontsize=8,
        )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ==============================================================================
# Session State Initialization
# ==============================================================================

provider_label, llm_callable = get_llm_callable_from_env()

if "papers" not in st.session_state:
    st.session_state.papers = get_sample_corpus()
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = run_full_pipeline(
        st.session_state.papers,
        llm_fn=llm_callable,
        mode_label=provider_label,
    )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "experiment_runs" not in st.session_state:
    st.session_state.experiment_runs = {}
if "benchmark_rows" not in st.session_state:
    st.session_state.benchmark_rows = None


# ==============================================================================
# Sidebar Controls
# ==============================================================================

with st.sidebar:
    st.markdown("### HypothesisForge")
    st.caption("Evidence-Grounded Research Gap & Hypothesis Workbench")
    st.divider()

    st.markdown("**Paper Source**")
    source_tab = st.radio(
        "Choose input method",
        ["Search by Topic", "Upload PDFs"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_tab == "Search by Topic":
        topic_query = st.text_input(
            "Research Topic",
            placeholder="e.g., Domain Generalization Vision Transformers",
        )
        max_papers = st.slider("Number of Papers", min_value=2, max_value=30, value=4)
        source_mode = st.selectbox(
            "Scholarly Sources",
            ["All (arXiv + OpenAlex + IEEE)", "arXiv + OpenAlex", "IEEE only", "arXiv only", "OpenAlex only"],
        )
        download_pdfs = st.checkbox("Download full PDFs when available", value=True)
        search_run_btn = st.button("Search & Run Analysis", type="primary", width="stretch")

        if search_run_btn:
            if not topic_query.strip():
                st.warning("Please enter a research topic first.")
            else:
                try:
                    with st.spinner(f"Searching {source_mode} for '{topic_query.strip()}'..."):
                        discovered = discover_papers(
                            topic=topic_query.strip(),
                            max_papers=max_papers,
                            source_mode=source_mode,
                        )
                    if not discovered:
                        st.warning("No papers found for that topic. Try broader keywords.")
                    else:
                        with st.spinner(f"Extracting text from {len(discovered)} papers & running analysis..."):
                            docs = fetch_paper_documents(discovered, download_pdfs=download_pdfs)
                            st.session_state.papers = docs
                            st.session_state.analysis_result = run_full_pipeline(
                                docs,
                                llm_fn=llm_callable,
                                mode_label=provider_label,
                                topic=topic_query.strip(),
                            )
                            st.session_state.chat_history = []
                        st.rerun()
                except Exception as exc:
                    st.error(f"Paper search failed: {exc}")
    else:
        uploaded_files = st.file_uploader(
            "Upload Research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload ML/DL research papers in PDF format (up to 200 MB per file).",
        )
        run_btn = st.button("Run Analysis", type="primary", width="stretch")

        if run_btn:
            if uploaded_files:
                loaded_docs: List[PaperDocument] = []
                for uf in uploaded_files:
                    doc = extract_pdf_bytes(uf.read(), uf.name)
                    loaded_docs.append(doc)
                st.session_state.papers = loaded_docs
            with st.spinner("Running 8-stage HypothesisForge pipeline..."):
                st.session_state.analysis_result = run_full_pipeline(
                    st.session_state.papers,
                    llm_fn=llm_callable,
                    mode_label=provider_label,
                )
                st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Active Corpus**")
    for p in st.session_state.papers:
        status_tag = "Abstract" if p.extraction_status == "abstract_only" else "Full Text"
        st.caption(f"• **{p.paper_id}** ({p.year}, {status_tag}) — {p.title[:38]}{'...' if len(p.title) > 38 else ''}")


# ==============================================================================
# Main Header & KPI Strip
# ==============================================================================

result: AnalysisPipelineResult = st.session_state.analysis_result
active_signature = compute_corpus_signature(st.session_state.papers, provider_label)
is_stale = result.corpus_signature != active_signature

st.markdown(
    f"""
    <div class="hf-header">
        <div class="hf-title">HypothesisForge Workbench</div>
        <p class="hf-subtitle">
            Traceable experimental extraction, cross-paper gap matrices, contradiction checks, and falsifiable hypothesis design.
            Engine: <strong>{esc(provider_label)}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_stale:
    st.warning("The paper corpus or engine mode changed since this report was generated. Run the analysis again to update these results.")

accepted_count = sum(1 for _, c in result.critic_reviews if c.verdict == "accept")
revise_count = sum(1 for _, c in result.critic_reviews if c.verdict == "revise")
llm_calls = sum(1 for a in result.audit_log if a.prompt_used)
rule_calls = sum(1 for a in result.audit_log if not a.prompt_used)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="hf-kpi-card">
            <div class="hf-kpi-label">Papers Indexed</div>
            <div class="hf-kpi-value">{len(result.papers)}</div>
            <div class="hf-kpi-note">{len(result.all_chunks)} searchable passages</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="hf-kpi-card">
            <div class="hf-kpi-label">Candidate Gaps</div>
            <div class="hf-kpi-value">{len(result.candidate_gaps)}</div>
            <div class="hf-kpi-note">Unobserved combinations</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="hf-kpi-card">
            <div class="hf-kpi-label">Hypotheses Reviewed</div>
            <div class="hf-kpi-value">{len(result.hypotheses)}</div>
            <div class="hf-kpi-note">{accepted_count} accepted · {revise_count} revise</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="hf-kpi-card">
            <div class="hf-kpi-label">Analyses Executed</div>
            <div class="hf-kpi-value">{len(result.audit_log)}</div>
            <div class="hf-kpi-note">LLM: {llm_calls} · Rule-Based: {rule_calls}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Main Workspace Tabs
# ==============================================================================

tabs = st.tabs([
    "Overview & Report",
    "Literature Records",
    "Gap Matrix",
    "Contradictions",
    "Hypotheses & Critic",
    "Experiments & Graph",
    "Scope Advisor",
    "Ask the Papers",
])


# --- TAB 1: Overview & Report (Prompt 07 + Audit Log) ---
with tabs[0]:
    st.subheader("Executive Report Summary")
    st.caption("Plain-language overview of what the papers cover, key candidate gaps, and what to double-check before running experiments.")

    if result.report_summary:
        paragraphs_html = "".join(f"<p style='margin-bottom:0.85rem; line-height:1.6;'>{esc(p)}</p>" for p in result.report_summary.paragraphs)
        st.markdown(
            f"""<div class="hf-card">
                <div style="margin-bottom:0.65rem;">
                    {pill(f"{result.report_summary.word_count} words", "blue")}
                    {pill(f"{len(result.report_summary.paragraphs)} paragraphs", "slate")}
                </div>
                {paragraphs_html}
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Technical Audit: Prompt Versions & Validation Traceability", expanded=False):
        audit_rows = [
            {
                "Stage": a.stage,
                "Item": a.item_id,
                "Prompt": a.prompt_name if a.prompt_used else "Prompt not used",
                "Version": a.prompt_version if a.prompt_used else "not used",
                "Retries": a.retries_used,
                "Mode": (
                    "Rule-Based (No LLM)"
                    if not a.prompt_used
                    else ("Rule-Based Fallback" if a.used_fallback else "Validated LLM")
                ),
            }
            for a in result.audit_log
        ]
        st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)

    with st.expander("Evaluation Harness (16-Paper Benchmark, Gap & Contradiction Precision, Ablation)", expanded=False):
        st.caption("Measures extraction precision, unsupported-claim rate, quote pass rate, gap precision, and contradiction precision.")
        if st.button("Run Full Evaluation Harness", key="run_bench_btn"):
            harness_report = run_full_evaluation_harness()
            st.session_state.benchmark_rows = {
                "table": [
                    {
                        "System": m.system_name,
                        "Extracted": m.total_extracted,
                        "Precision (%)": m.precision,
                        "Recall (%)": m.recall,
                        "Unsupported Claim Rate (%)": m.unsupported_claim_rate,
                        "Quote Pass Rate (%)": m.quote_pass_rate,
                    }
                    for m in harness_report.extraction_rows
                ],
                "gap_precision": harness_report.gap_precision,
                "contradiction_precision": harness_report.contradiction_precision,
            }
        if st.session_state.benchmark_rows:
            bdata = st.session_state.benchmark_rows
            bc1, bc2 = st.columns(2)
            bc1.metric("Gap Matrix Precision", f"{bdata['gap_precision']:.1f}%")
            bc2.metric("Contradiction Precision", f"{bdata['contradiction_precision']:.1f}%")
            st.dataframe(pd.DataFrame(bdata["table"]), width="stretch", hide_index=True)

    export_payload = {
        "report_summary": result.report_summary.model_dump() if result.report_summary else None,
        "literature_records": {k: v.model_dump() for k, v in result.literature_records.items()},
        "candidate_gaps": result.candidate_gaps,
        "gap_assessments": [{"gap": g, **a.model_dump()} for g, a in result.gap_assessments],
        "contradictions": [{"pair": p, **c.model_dump()} for p, c in result.contradictions],
        "hypotheses": [{"gap": g, **h.model_dump()} for g, h in result.hypotheses],
        "critic_reviews": [{"hypothesis": h, **c.model_dump()} for h, c in result.critic_reviews],
        "experiment_plans": [{"hypothesis": h, **e.model_dump()} for h, e in result.experiment_plans],
    }
    st.download_button(
        label="Download Full Analysis Report (JSON)",
        data=json.dumps(export_payload, indent=2),
        file_name="hypothesisforge_report.json",
        mime="application/json",
    )


# --- TAB 2: Literature Records (Prompt 01) ---
with tabs[1]:
    st.subheader("Structured Literature Records")
    st.caption("Key experimental details extracted from each paper, including models, datasets, numeric results, and supporting quotes.")

    paper_map = {f"{p.paper_id} — {p.title}": p for p in result.papers}
    selected_label = st.selectbox("Select Paper", list(paper_map.keys()))
    sel_paper = paper_map[selected_label]
    rec = result.literature_records[sel_paper.paper_id]

    meta_pills = (
        verdict_pill(f"Confidence: {rec.extraction_confidence}")
        + pill(f"Year: {sel_paper.year}", "slate")
        + pill(f"Pages: {len(sel_paper.pages)}", "slate")
    )
    st.markdown(f'<div class="hf-card">{meta_pills}<br/><br/>'
                f'<strong>Tasks:</strong> {esc(", ".join(rec.tasks) or "not stated")}<br/>'
                f'<strong>Architectures:</strong> {esc(", ".join(rec.architectures) or "not stated")}<br/>'
                f'<strong>Datasets:</strong> {esc(", ".join(rec.datasets) or "not stated")}<br/>'
                f'<strong>Training Strategies:</strong> {esc(", ".join(rec.training_strategies) or "not stated")}<br/>'
                f'<strong>Shift Settings:</strong> {esc(", ".join(rec.shift_settings) or "not stated")}<br/>'
                f'<strong>Evaluation Protocol:</strong> {esc(rec.evaluation_protocol)}'
                f'</div>', unsafe_allow_html=True)

    st.markdown("**Reported Numeric Results**")
    if rec.reported_results:
        res_df = pd.DataFrame([
            {
                "Value": r.value,
                "Metric": r.metric,
                "Dataset": r.dataset,
                "Setting": r.setting,
                "Method": r.method,
                "Page": r.page,
                "Verbatim Quote": r.quote,
            }
            for r in rec.reported_results
        ])
        st.dataframe(res_df, width="stretch", hide_index=True)
    else:
        st.info("No numeric results extracted with full context.")

    col_f, col_l = st.columns(2)
    with col_f:
        st.markdown("**Findings**")
        if rec.findings:
            for f_item in rec.findings:
                st.markdown(
                    f"""<div class="hf-card-muted">
                        <div>{pill(f"Page {f_item.page}", "blue")}{pill(f_item.section, "slate")}</div>
                        <div style="margin-top:0.4rem;">{esc(f_item.text)}</div>
                        <div class="hf-quote">"{esc(f_item.quote)}"</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No separate findings extracted in current mode.")

    with col_l:
        st.markdown("**Limitations**")
        if rec.limitations:
            for l_item in rec.limitations:
                st.markdown(
                    f"""<div class="hf-card-muted">
                        <div>{verdict_pill(l_item.origin)}{pill(f"Page {l_item.page}", "blue")}</div>
                        <div style="margin-top:0.4rem;">{esc(l_item.text)}</div>
                        <div class="hf-quote">"{esc(l_item.quote)}"</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No separate limitations extracted in current mode.")


# --- TAB 3: Gap Matrix & Relevance (Prompt 02) ---
with tabs[2]:
    st.subheader("What Has & Has Not Been Tested (Gap Matrix)")
    st.markdown(
        """<div class="hf-card-muted">
            <strong>How to read this:</strong> This table compares which <strong>AI models (Architectures)</strong> were tested on which <strong>benchmarks (Datasets)</strong> across the papers.
            <ul style="margin:0.35rem 0 0 1.2rem; font-size:0.88rem; color:#334155;">
                <li><strong>Observed in corpus:</strong> Already tested in at least one of the papers we read.</li>
                <li><strong>Candidate gap (unobserved):</strong> A model + dataset combination that did not appear in these papers. This does not prove nobody in the world has ever tested it—it is simply a promising candidate to verify.</li>
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )

    if result.matrix_rows:
        st.dataframe(pd.DataFrame(result.matrix_rows), width="stretch", hide_index=True)

    st.subheader("Topic & Scope Advisor")
    st.caption("Short scope interview: recommends 3–5 mini or major options grounded strictly in the Gap Matrix above.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        team_size = st.number_input("Team Size", min_value=1, max_value=6, value=2)
    with sc2:
        timeline_weeks = st.slider("Timeline (weeks)", min_value=2, max_value=16, value=6)
    with sc3:
        compute_budget = st.selectbox("Compute Budget", ["CPU / Colab Free", "Single GPU", "Multi-GPU"])
    with sc4:
        goal_scope = st.selectbox("Target Scope", ["Mini Project", "Major Project", "Any"])

    interview = ScopeInterview(
        team_size=int(team_size),
        timeline_weeks=int(timeline_weeks),
        compute_budget=compute_budget,
        goal=goal_scope,
    )
    scoped_recs = recommend_scoped_topics(interview, result, max_options=5)
    paper_title_map = {p.paper_id: p.title for p in result.papers}

    def _explain_gap_simply(gap_label_str: str) -> str:
        if " on " not in gap_label_str:
            return f"This combination ({gap_label_str}) did not appear in the {len(result.papers)} loaded papers."
        arch_part, ds_part = [x.strip() for x in gap_label_str.split(" on ", 1)]
        arch_papers = [
            pid for pid, rec in result.literature_records.items()
            if any(arch_part.lower() in a.lower() or a.lower() in arch_part.lower() for a in rec.architectures)
        ]
        ds_papers = [
            pid for pid, rec in result.literature_records.items()
            if any(ds_part.lower() in d.lower() or d.lower() in ds_part.lower() for d in rec.datasets)
        ]
        a_txt = ", ".join(arch_papers[:3]) if arch_papers else "some papers"
        d_txt = ", ".join(ds_papers[:3]) if ds_papers else "other papers"
        return (
            f"In the {len(result.papers)} papers searched, {arch_part} appears in {a_txt} and "
            f"{ds_part} appears in {d_txt}, but none of them tested {arch_part} on {ds_part} together."
        )

    if not scoped_recs:
        st.info("No candidate gaps in the matrix yet to scope.")
    for rec_item in scoped_recs:
        scope_variant = "green" if rec_item.scope_label == "mini" else "blue"
        paper_pills = "".join(
            pill(f"{pid}: {paper_title_map.get(pid, '')[:38]}", "slate")
            for pid in rec_item.motivating_papers
        )
        risk_items = "".join(f"<li>{esc(rk)}</li>" for rk in rec_item.risks)
        easy_gap = _explain_gap_simply(rec_item.gap_label)
        st.markdown(
            f"""<div class="hf-card">
                <div class="hf-card-title">{esc(rec_item.gap_label)} {pill(rec_item.scope_label.upper(), scope_variant)}</div>
                <div style="background:#f8fafc; border-left:3px solid #2563eb; padding:0.45rem 0.7rem; margin-bottom:0.5rem; font-size:0.89rem; color:#1e293b;">
                    <strong>Easy Explanation:</strong> {esc(easy_gap)}
                </div>
                <div style="margin-bottom:0.45rem;"><strong>Related Papers:</strong> {paper_pills}</div>
                <div style="font-size:0.88rem; color:#334155;"><strong>Required Compute:</strong> {esc(rec_item.required_compute)}</div>
                <div style="font-size:0.88rem; color:#334155; margin-top:0.25rem;"><strong>Scope Fit:</strong> {esc(rec_item.fit_reason)}</div>
                <div style="font-size:0.82rem; font-weight:600; color:#475569; margin-top:0.4rem;">Risks:</div>
                <ul style="margin:0.2rem 0 0 1.2rem; color:#334155; font-size:0.85rem;">{risk_items}</ul>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Why These Candidate Gaps Matter")
    st.caption("Plain-language assessment of each untested model-and-dataset combination found in this set of papers.")

    if not result.gap_assessments:
        st.info("No candidate gaps detected in the current corpus.")
    for gap_label, assessment in result.gap_assessments:
        qs_html = "".join(f"<li>{esc(q)}</li>" for q in assessment.questions_to_check)
        easy_gap = _explain_gap_simply(gap_label)
        st.markdown(
            f"""<div class="hf-card">
                <div class="hf-card-title">{esc(gap_label)}</div>
                <div style="margin-bottom:0.55rem;">
                    {verdict_pill(f"Rating: {assessment.rating}")}
                    {pill(f"Absence: {assessment.absence_explanation}", "slate")}
                </div>
                <div style="background:#f8fafc; border-left:3px solid #2563eb; padding:0.45rem 0.7rem; margin-bottom:0.5rem; font-size:0.89rem; color:#1e293b;">
                    <strong>What's Missing:</strong> {esc(easy_gap)}
                </div>
                <p style="margin-bottom:0.5rem; color:#334155;">{esc(assessment.reason)}</p>
                <div style="font-size:0.85rem; font-weight:600; color:#475569;">Questions to Check:</div>
                <ul style="margin:0.25rem 0 0 1.2rem; color:#334155; font-size:0.88rem;">{qs_html}</ul>
            </div>""",
            unsafe_allow_html=True,
        )


# --- TAB 4: Contradictions (Prompt 03) ---
with tabs[3]:
    st.subheader("Cross-Paper Contradiction Judgement")
    st.markdown(
        """<div class="hf-card-muted">
            <strong>How to read this:</strong> Two papers having different accuracy numbers does <em>not</em> mean one paper is wrong.
            <ul style="margin:0.35rem 0 0 1.2rem; font-size:0.88rem; color:#334155;">
                <li><strong>Non-comparable evidence:</strong> The papers tested on different datasets or tasks, so their numbers cannot be compared directly.</li>
                <li><strong>Insufficient information:</strong> Key testing details (like train/test split or protocol) are missing, so more checking is needed.</li>
                <li><strong>Apparent agreement / disagreement:</strong> Both papers used comparable settings and either agree or conflict.</li>
            </ul>
        </div>""",
        unsafe_allow_html=True,
    )

    if not result.contradictions:
        st.info("Need at least two papers with numeric results to compare pairs.")
    import re as _re_ui
    paper_title_lookup = {p.paper_id: p.title for p in result.papers}
    for pair_title, cj in result.contradictions:
        cond_html = "".join(f"<li>{esc(c)}</li>" for c in cj.conditions_to_verify)
        conf_pct = int(round(cj.confidence_score * 100))
        conf_variant = "green" if conf_pct >= 75 else ("amber" if conf_pct >= 50 else "red")
        factor_pills = "".join(pill(f"Factor: {f}", "blue") for f in cj.conflict_factors)

        pids = _re_ui.findall(r"\bP-[A-Z0-9]+\b", pair_title)
        pa_id = pids[0] if len(pids) >= 1 else "Paper A"
        pb_id = pids[1] if len(pids) >= 2 else "Paper B"
        pa_title = paper_title_lookup.get(pa_id, pa_id)
        pb_title = paper_title_lookup.get(pb_id, pb_id)

        rec_a = result.literature_records.get(pa_id)
        rec_b = result.literature_records.get(pb_id)
        res_a = rec_a.reported_results[0] if rec_a and rec_a.reported_results else None
        res_b = rec_b.reported_results[0] if rec_b and rec_b.reported_results else None

        def _fmt_side(pid_str: str, title_str: str, rec_obj, res_obj) -> str:
            ds = (res_obj.dataset if res_obj and res_obj.dataset else None) or (", ".join(rec_obj.datasets[:2]) if rec_obj and rec_obj.datasets else "Not stated")
            arch = (getattr(res_obj, "model", None) if res_obj else None) or (", ".join(rec_obj.architectures[:2]) if rec_obj and rec_obj.architectures else "Not stated")
            met = res_obj.metric if res_obj and res_obj.metric else "Not stated"
            val = str(res_obj.value) if res_obj and res_obj.value is not None else "Not stated"
            spl = str(rec_obj.evaluation_protocol)[:60] if rec_obj and rec_obj.evaluation_protocol else "Not stated"
            return (
                f"<strong>{esc(pid_str)}</strong> — {esc(title_str[:65])}<br>"
                f"<span style='font-size:0.83rem; color:#475569;'>"
                f"Model: <b>{esc(arch)}</b> · Dataset: <b>{esc(ds)}</b> · Metric: <b>{esc(met)}</b> · Value: <b>{esc(val)}</b> · Split: <b>{esc(spl)}</b>"
                f"</span>"
            )

        side_a_html = _fmt_side(pa_id, pa_title, rec_a, res_a)
        side_b_html = _fmt_side(pb_id, pb_title, rec_b, res_b)

        # Plain-English issue summary
        cls_low = cj.classification.lower()
        factors_txt = ", ".join(cj.conflict_factors) if cj.conflict_factors else "dataset, metric, or split"
        scale_note = ""
        if res_a and res_b and res_a.value is not None and res_b.value is not None:
            try:
                va, vb = float(res_a.value), float(res_b.value)
                if (va > 1.0 and vb <= 1.0) or (vb > 1.0 and va <= 1.0):
                    scale_note = " Note: one number is > 1 (likely %) while the other is <= 1 (likely 0-1 decimal or a different metric)."
            except Exception:
                pass

        if "insufficient" in cls_low:
            issue_line = f"Issue: Cannot tell if these conflict yet because {factors_txt} details are missing in one or both papers.{scale_note}"
            issue_bg, issue_border = "#fffbeb", "#d97706"
        elif "non-comparable" in cls_low:
            issue_line = f"Issue: Apples-to-oranges comparison — the papers differ on {factors_txt}, so the numbers cannot be compared directly.{scale_note}"
            issue_bg, issue_border = "#eff6ff", "#2563eb"
        elif "disagreement" in cls_low:
            issue_line = f"Issue: Real conflict candidate — both papers used comparable settings, yet reported different results.{scale_note}"
            issue_bg, issue_border = "#fef2f2", "#dc2626"
        else:
            issue_line = f"No conflict: Both papers report consistent results under comparable settings.{scale_note}"
            issue_bg, issue_border = "#f0fdf4", "#16a34a"

        st.markdown(
            f"""<div class="hf-card">
                <div class="hf-card-title">{esc(pair_title)}</div>
                <div style="margin-bottom:0.55rem;">
                    {verdict_pill(cj.classification)}
                    {pill(f"Confidence: {conf_pct}%", conf_variant)}
                    {factor_pills}
                </div>
                <div style="background:{issue_bg}; border-left:3px solid {issue_border}; padding:0.45rem 0.7rem; margin-bottom:0.55rem; font-size:0.89rem; color:#1e293b;">
                    <strong>{esc(issue_line)}</strong>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.55rem 0.75rem; margin-bottom:0.55rem; font-size:0.86rem; color:#1e293b;">
                    <div style="margin-bottom:0.35rem;">{side_a_html}</div>
                    <div>{side_b_html}</div>
                </div>
                <p style="margin-bottom:0.5rem; color:#334155;">{esc(cj.explanation)}</p>
                <div style="font-size:0.85rem; font-weight:600; color:#475569;">Conditions to Verify:</div>
                <ul style="margin:0.25rem 0 0 1.2rem; color:#334155; font-size:0.88rem;">{cond_html}</ul>
            </div>""",
            unsafe_allow_html=True,
        )


# --- TAB 5: Hypotheses & Critic (Prompts 04 & 05) ---
with tabs[4]:
    st.subheader("Testable Hypotheses & Critical Review")
    st.caption("Specific, testable predictions grounded in the papers, followed by an independent critique checking for weaknesses or missing evidence.")

    if not result.hypotheses:
        st.info("No candidate gaps available to generate hypotheses.")

    critic_by_stmt = {stmt: crit for stmt, crit in result.critic_reviews}

    for idx, (gap_label, hyp) in enumerate(result.hypotheses, start=1):
        if hyp.decision == "insufficient_evidence":
            st.markdown(
                f"""<div class="hf-card">
                    <div class="hf-card-title">H{idx}: {esc(gap_label)}</div>
                    {verdict_pill("insufficient_evidence")}
                    <p style="margin-top:0.5rem; color:#334155;">{esc(hyp.reason)}</p>
                </div>""",
                unsafe_allow_html=True,
            )
            continue

        crit = critic_by_stmt.get(hyp.statement or gap_label)
        assump_html = "".join(f"<li>{esc(a)}</li>" for a in (hyp.assumptions or []))
        cited_pills = "".join(pill(pid, "blue") for pid in (hyp.cited_paper_ids or []))

        crit_html = ""
        if crit:
            conf_html = "".join(f"<li>{esc(x)}</li>" for x in crit.alternative_explanations)
            miss_html = "".join(f"<li>{esc(x)}</li>" for x in crit.missing_evidence)
            conf_block = f'<div style="font-size:0.82rem; font-weight:600; color:#475569;">Alternative Explanations / Confounders:</div><ul style="margin:0.2rem 0 0.4rem 1.2rem; font-size:0.85rem;">{conf_html}</ul>' if conf_html else ""
            miss_block = f'<div style="font-size:0.82rem; font-weight:600; color:#475569;">Missing Evidence:</div><ul style="margin:0.2rem 0 0 1.2rem; font-size:0.85rem;">{miss_html}</ul>' if miss_html else ""
            crit_html = (
                f'<div class="hf-card-muted" style="margin-top:0.75rem; margin-bottom:0;">'
                f'<div style="font-weight:600; margin-bottom:0.35rem;">Scientific Critic Review {verdict_pill(crit.verdict)}</div>'
                f'<p style="font-size:0.9rem; color:#334155; margin-bottom:0.45rem;">{esc(crit.rationale)}</p>'
                f'{conf_block}{miss_block}'
                f'</div>'
            )

        card_html = (
            f'<div class="hf-card">'
            f'<div class="hf-card-title">H{idx}: {esc(gap_label)}</div>'
            f'<div style="margin-bottom:0.55rem;">'
            f'{pill(f"Direction: {hyp.expected_direction}", "slate")}'
            f'{pill(f"Metric: {hyp.metric}", "slate")}'
            f'{cited_pills}'
            f'</div>'
            f'<div class="hf-quote"><strong>Statement:</strong> {esc(hyp.statement)}</div>'
            f'<div style="font-size:0.88rem; color:#334155;"><strong>Baseline:</strong> {esc(hyp.baseline)}</div>'
            f'<div style="font-size:0.88rem; color:#334155; margin-top:0.25rem;"><strong>Rationale:</strong> {esc(hyp.reason)}</div>'
            f'<div style="font-size:0.82rem; font-weight:600; color:#475569; margin-top:0.45rem;">Assumptions:</div>'
            f'<ul style="margin:0.2rem 0 0 1.2rem; color:#334155; font-size:0.85rem;">{assump_html}</ul>'
            f'{crit_html}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


# --- TAB 6: Experiment Plans & Knowledge Graph (Prompt 06 + NetworkX) ---
with tabs[5]:
    st.subheader("Step-by-Step Experiment Plans")
    st.caption("Fair, repeatable experiment checklists showing datasets, baselines, random seed runs, and potential pitfalls.")

    if not result.experiment_plans:
        st.info("No non-rejected hypotheses available for experiment design.")
    for idx, (hyp_text, plan) in enumerate(result.experiment_plans, start=1):
        with st.expander(f"Experiment Plan E{idx}: {plan.research_question}", expanded=(idx == 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Independent Variable:** {plan.independent_variable}")
                st.markdown(f"**Dependent Variable:** {plan.dependent_variable}")
                st.markdown(f"**Dataset & Split:** {plan.dataset_and_split}")
                st.markdown(f"**Baselines:** {', '.join(plan.baselines)}")
                st.markdown(f"**Controls:** {', '.join(plan.controls)}")
            with c2:
                st.markdown(f"**Metrics:** {', '.join(plan.metrics)}")
                st.markdown(f"**Seeds & Runs:** {plan.seeds_and_runs}")
                st.markdown(f"**Statistics:** {plan.statistics}")
                st.markdown(f"**Compute Estimate:** {plan.compute_estimate}")

            st.markdown("**Ordered Procedure (max 7 steps)**")
            for step in plan.procedure:
                st.markdown(f"- {step}")

            st.markdown(f"**Expected Observations:** {plan.expected_observations}")
            if plan.threats_to_validity:
                st.markdown("**Threats to Validity:**")
                for t in plan.threats_to_validity:
                    st.markdown(f"- {t}")

            st.divider()
            col_ex1, col_ex2 = st.columns([1, 2])
            with col_ex1:
                if st.button(f"Run Small-Scale Experiment E{idx}", key=f"exec_exp_{idx}", type="primary"):
                    with st.spinner("Executing generated 5-seed benchmark script..."):
                        st.session_state.experiment_runs[idx] = execute_experiment_plan(plan, n_seeds=5)
            with col_ex2:
                st.caption("Generates and executes a runnable scikit-learn script (shifted Digits proxy) across 5 seeds with paired t-test.")

            if idx in st.session_state.experiment_runs:
                exec_res = st.session_state.experiment_runs[idx]
                out_variant = "green" if exec_res.outcome == "confirmed" else ("red" if exec_res.outcome == "refuted" else "amber")
                st.markdown(
                    f"""<div class="hf-card-muted">
                        <strong>Execution Result:</strong> {pill(exec_res.outcome.upper(), out_variant)}
                        {pill(f"Delta: {exec_res.delta:+.2f}%", "blue")}
                        {pill(f"p-value: {exec_res.p_value:.4f}", "slate")}
                        <p style="margin:0.45rem 0 0.2rem 0; font-size:0.88rem; color:#334155;">{esc(exec_res.summary)}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                seed_df = pd.DataFrame({
                    "Seed": exec_res.seeds,
                    "Baseline Accuracy (%)": exec_res.baseline_scores,
                    "Proposed Accuracy (%)": exec_res.proposed_scores,
                })
                st.dataframe(seed_df, width="stretch", hide_index=True)

            with st.expander(f"View Generated Python Script (E{idx})", expanded=False):
                script_code = generate_experiment_script(plan, n_seeds=5)
                st.code(script_code, language="python")

    st.divider()
    st.subheader("Corpus Knowledge Graph")
    st.caption("NetworkX graph linking Papers, Architectures (merged ViT/ViT-B/16), Datasets, Methods, and Hypotheses.")
    kg_col1, kg_col2 = st.columns([3, 1])
    with kg_col1:
        visible_types = st.multiselect(
            "Visible Node Types",
            ["Paper", "Architecture", "Dataset", "Method", "Hypothesis"],
            default=["Paper", "Architecture", "Dataset", "Hypothesis"],
        )
    with kg_col2:
        show_edge_labels = st.checkbox("Show edge labels", value=False)
    fig = render_knowledge_graph(
        result.knowledge_graph,
        visible_types=visible_types,
        show_edge_labels=show_edge_labels,
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# --- TAB 7: Topic & Scope Advisor ---
with tabs[6]:
    st.subheader("Topic & Scope Advisor")
    st.caption("Type any topic on your mind to search related papers and get 3–5 concrete Mini or Major project options grounded in the Gap Matrix.")

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        mind_topic = st.text_input(
            "What topic or idea is on your mind?",
            placeholder="e.g., brain tumor MRI classification, domain generalization, time series forecasting...",
            key="advisor_mind_topic",
        )
    with tc2:
        st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
        advisor_search_btn = st.button("Search & Suggest Topics", type="primary", width="stretch", key="advisor_search_btn")

    if advisor_search_btn and mind_topic.strip():
        with st.spinner(f"Searching papers for '{mind_topic.strip()}' and extracting research topics..."):
            found_meta = discover_papers(
                topic=mind_topic.strip(),
                max_papers=12,
                source_mode="All (arXiv + OpenAlex + IEEE)",
            )
            if not found_meta:
                st.warning("No papers found for that topic. Try broader keywords.")
            else:
                fetched_docs = fetch_paper_documents(found_meta, download_pdfs=False)
                if fetched_docs:
                    st.session_state.papers = fetched_docs
                    st.session_state.analysis_result = run_full_pipeline(
                        fetched_docs,
                        llm_fn=llm_callable,
                        mode_label=provider_label,
                        topic=mind_topic.strip(),
                    )
                    st.session_state.chat_history = []
                    st.rerun()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        target_scope = st.selectbox("Target Project Scale", ["Any", "Mini Project", "Major Project"], index=0)
    with sc2:
        compute_budget = st.selectbox("Available Compute", ["CPU / Colab Free", "Single GPU", "Multi-GPU"], index=1)
    with sc3:
        timeline_weeks = st.slider("Available Timeline (weeks)", min_value=2, max_value=16, value=6)

    # Prepare grounded inputs for Prompt 11 (Topic & Scope Recommender)
    enriched_gaps = []
    for g in result.candidate_gaps:
        arch_l = g["architecture"].lower()
        ds_l = g["dataset"].lower()
        sup_ids = [
            pid for pid, rec in result.literature_records.items()
            if any(arch_l in a.lower() or a.lower() in arch_l for a in rec.architectures)
            or ds_l in [d.lower() for d in rec.datasets]
        ] or [p.paper_id for p in result.papers[:1]]
        enriched_gaps.append({
            "gap_label": g["gap_label"],
            "architecture": g["architecture"],
            "dataset": g["dataset"],
            "status": g.get("status", "not_found_in_corpus"),
            "supporting_paper_ids": sup_ids,
        })

    import re as _re_con
    con_inputs = []
    for pair_title, cj in result.contradictions:
        pids_found = _re_con.findall(r"\bP-[A-Z0-9]+\b", pair_title)
        pa = pids_found[0] if len(pids_found) >= 1 else ""
        pb = pids_found[1] if len(pids_found) >= 2 else ""
        rec_a = result.literature_records.get(pa)
        ds_name = (rec_a.datasets[0] if rec_a and rec_a.datasets else "") or "the shared task"
        con_inputs.append({
            "paper_a_id": pa,
            "paper_b_id": pb,
            "dataset": ds_name,
            "pair": pair_title,
            "classification": cj.classification,
            "confidence_score": cj.confidence_score,
            "conflict_factors": cj.conflict_factors,
            "explanation": cj.explanation,
        })
    scope_criteria_str = f"target_scale={target_scope}, compute_budget={compute_budget}, available_weeks={timeline_weeks}"
    allowed_pids = [p.paper_id for p in result.papers]
    allowed_ds = sorted({d for rec in result.literature_records.values() for d in rec.datasets})

    rec_exec = PromptRunner().run_topic_recommendation(
        profile={"field_of_interest": mind_topic.strip() or "loaded papers", "available_weeks": timeline_weeks, "compute_access": compute_budget, "project_level": target_scope},
        gaps=enriched_gaps,
        contradictions=con_inputs,
        scope_criteria=scope_criteria_str,
        n_papers=len(result.papers),
        allowed_paper_ids=allowed_pids,
        allowed_datasets=allowed_ds,
        llm_fn=llm_callable,
    )
    topic_out = rec_exec.data

    if topic_out.warnings:
        for w in topic_out.warnings:
            st.warning(w)

    if not topic_out.options:
        st.info("No candidate gaps or contradictions in the current corpus to scope. Search a topic or upload papers with explicit models and datasets.")
    else:
        for idx, opt in enumerate(topic_out.options, start=1):
            scope_variant = "green" if opt.scope == "mini" else "blue"
            ev_html = ", ".join(esc(e) for e in opt.evidence_ids)
            read_html = ", ".join(esc(p) for p in opt.papers_to_read)
            ds_html = ", ".join(esc(d) for d in opt.datasets)
            risk_html = "".join(f"<li>{esc(r)}</li>" for r in opt.risks)
            st.markdown(
                f"""<div class="hf-card">
                    <div class="hf-card-title">Option {idx}: {esc(opt.title)}</div>
                    <div style="margin-bottom:0.55rem;">
                        {pill(opt.scope.upper(), scope_variant)}
                        {pill(f"{opt.effort_weeks} weeks", "slate")}
                        {pill(f"Datasets: {ds_html}", "slate")}
                    </div>
                    <div style="font-size:0.9rem; color:#1e293b; margin-bottom:0.4rem;">{esc(opt.reason)}</div>
                    <div style="font-size:0.86rem; color:#334155;"><strong>Compute Needed:</strong> {esc(opt.compute_needed)}</div>
                    <div style="font-size:0.86rem; color:#334155;"><strong>Traceable Evidence IDs:</strong> {ev_html} | <strong>Papers to Read:</strong> {read_html}</div>
                    <div style="font-size:0.84rem; font-weight:600; color:#475569; margin-top:0.45rem;">Risks to Check:</div>
                    <ul style="margin:0.2rem 0 0 1.2rem; color:#334155; font-size:0.86rem;">{risk_html}</ul>
                </div>""",
                unsafe_allow_html=True,
            )


# --- TAB 8: Ask the Papers (Prompt 08) ---
with tabs[7]:
    st.subheader("Ask the Papers")
    st.caption("Ask any question about the loaded papers. Every answer cites numbered [n] passages directly from the source text.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("passages"):
                with st.expander("View Retrieved Passages"):
                    st.text(msg["passages"])

    user_question = st.chat_input("Ask about the papers or how to build on a gap (e.g., How do I build with any of the gaps?)")
    if user_question:
        ans_text, passages_str, mode_note = answer_workspace_question(
            question=user_question,
            result=result,
            llm_fn=llm_callable,
            chat_history=st.session_state.chat_history,
        )
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"{ans_text}\n\n*{mode_note}*",
            "passages": passages_str,
        })
        st.rerun()
