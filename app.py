import streamlit as st
import os
import json
import uuid
import shutil
import numpy as np
import time
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from pipeline.extractor import extract_paper
from pipeline.chunker import chunk_papers
from pipeline.embedder import load_embedding_model, embed_chunks
from pipeline.clusterer import cluster_embeddings
from pipeline.summariser import configure_gemini, summarise_all_clusters
from pipeline.gap_analyzer import perform_gap_analysis
from export_utils import export_markdown, export_pdf, export_docx

# ------------- Gemini API key handling (works both locally and on Streamlit Cloud) -------------
try:
    # If running on Streamlit Cloud, secrets are available
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback: load from .env file for local development
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("No GEMINI_API_KEY found. Set it in .env or Streamlit secrets.")
        st.stop()
os.environ["GEMINI_API_KEY"] = api_key

# Page config (supports dark/light mode via user toggle)
st.set_page_config(page_title="Research Gap Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- CSS for light/dark mode ---
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #e0e0e0; }
        .stExpander { background-color: #1a1c23; border: 1px solid #333; }
        .css-1d391kg, .css-1wrcr25 { background-color: #1a1c23; }
        .metric-card { background-color: #1a1c23; padding: 10px; border-radius: 8px; margin: 5px; }
        h1, h2, h3, h4, h5, h6 { color: #f0f0f0; }
        .stDownloadButton button { background-color: #4CAF50; color: white; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin: 5px; }
        .stDownloadButton button { background-color: #4CAF50; color: white; }
        </style>
        """, unsafe_allow_html=True)

# --- Session state ---
if 'stage' not in st.session_state:
    st.session_state['stage'] = 0
if 'papers' not in st.session_state:
    st.session_state['papers'] = []
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'clusters' not in st.session_state:
    st.session_state['clusters'] = []
if 'summaries' not in st.session_state:
    st.session_state['summaries'] = []
if 'gap_report' not in st.session_state:
    st.session_state['gap_report'] = None
if 'topic' not in st.session_state:
    st.session_state['topic'] = ""
if 'embed_model' not in st.session_state:
    st.session_state['embed_model'] = None
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'error' not in st.session_state:
    st.session_state['error'] = None
if 'failure_log' not in st.session_state:
    st.session_state['failure_log'] = []
if 'theme' not in st.session_state:
    st.session_state['theme'] = "Light"

# --- Theme toggle in sidebar ---
with st.sidebar:
    st.write("### Settings")
    new_theme = st.radio("Mode", ["Light", "Dark"], index=0 if st.session_state['theme']=="Light" else 1)
    if new_theme != st.session_state['theme']:
        st.session_state['theme'] = new_theme
        st.rerun()
    st.write("---")
    st.caption("Research Gap Analyzer v1.0")

apply_theme(st.session_state['theme'])

# --- Helper: session dirs ---
def session_dir():
    d = os.path.join("sessions", st.session_state.session_id)
    os.makedirs(d, exist_ok=True)
    return d

def upload_dir():
    d = os.path.join("uploads", st.session_state.session_id)
    os.makedirs(d, exist_ok=True)
    return d

# --- Stage runners with failure logging ---
def run_stage_1():
    uploaded_files = st.session_state.get('uploaded_files', [])
    papers = []
    failure_log = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        filepath = os.path.join(upload_dir(), filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paper = extract_paper(filepath, filename)
        if not paper['readable']:
            failure_log.append(f"Unreadable PDF: {filename} (likely scanned or too short)")
        papers.append(paper)
    st.session_state['papers'] = papers
    st.session_state['failure_log'].extend(failure_log)
    with open(os.path.join(session_dir(), "papers.json"), "w") as f:
        json.dump(papers, f, indent=2)
    return len(papers)

def run_stage_2():
    papers = st.session_state['papers']
    chunks = chunk_papers(papers)
    if len(chunks) < 4:
        st.session_state['failure_log'].append("Too few chunks for meaningful clustering. Need at least 4 chunks.")
    st.session_state['chunks'] = chunks
    with open(os.path.join(session_dir(), "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)
    return len(chunks)

def run_stage_3():
    chunks = st.session_state['chunks']
    if not chunks:
        return 0
    if st.session_state['embed_model'] is None:
        with st.spinner("Loading embedding model..."):
            model = load_embedding_model()
            st.session_state['embed_model'] = model
    else:
        model = st.session_state['embed_model']
    embeddings = embed_chunks(chunks, model)
    st.session_state['embeddings'] = embeddings
    np.save(os.path.join(session_dir(), "embeddings.npy"), embeddings)
    return embeddings.shape[0]

def run_stage_4():
    embeddings = st.session_state['embeddings']
    chunks = st.session_state['chunks']
    papers = st.session_state['papers']
    n_readable = len([p for p in papers if p['readable']])
    if embeddings is None or embeddings.shape[0] == 0:
        return []
    try:
        clusters = cluster_embeddings(embeddings, chunks, n_readable)
    except Exception as e:
        st.session_state['failure_log'].append(f"Clustering failed: {str(e)}")
        return []
    st.session_state['clusters'] = clusters
    with open(os.path.join(session_dir(), "clusters.json"), "w") as f:
        json.dump(clusters, f, indent=2)
    return len(clusters)

def run_stage_5():
    configure_gemini()
    clusters = st.session_state['clusters']
    if not clusters:
        return []
    try:
        summaries = summarise_all_clusters(clusters)
    except Exception as e:
        st.session_state['failure_log'].append(f"Summarisation error: {str(e)}")
        return []
    for s in summaries:
        if "failed" in s['summary'].lower():
            st.session_state['failure_log'].append(f"Theme {s['cluster_id']} summary may be incomplete.")
    st.session_state['summaries'] = summaries
    with open(os.path.join(session_dir(), "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    return len(summaries)

def run_stage_6():
    configure_gemini()
    summaries = st.session_state['summaries']
    papers = st.session_state['papers']
    n_readable = len([p for p in papers if p['readable']])
    topic = st.session_state['topic']
    try:
        gap_report = perform_gap_analysis(summaries, topic, n_readable)
        if 'error' in gap_report:
            st.session_state['failure_log'].append(f"Gap analysis error: {gap_report['error']}")
    except Exception as e:
        gap_report = {"error": str(e)}
        st.session_state['failure_log'].append(f"Gap analysis exception: {str(e)}")
    st.session_state['gap_report'] = gap_report
    with open(os.path.join(session_dir(), "gap_report.json"), "w") as f:
        json.dump(gap_report, f, indent=2)
    return gap_report

def run_full_pipeline():
    progress = st.progress(0)
    status = st.empty()

    status.text("Extracting text from PDFs...")
    n = run_stage_1()
    progress.progress(1/6)

    status.text("Chunking papers...")
    n_chunks = run_stage_2()
    progress.progress(2/6)

    status.text("Generating embeddings...")
    n_emb = run_stage_3()
    progress.progress(3/6)

    status.text("Clustering...")
    n_clusters = run_stage_4()
    progress.progress(4/6)

    status.text("Summarising clusters with LLM...")
    n_summaries = run_stage_5()
    progress.progress(5/6)

    status.text("Identifying research gaps...")
    gap_report = run_stage_6()
    progress.progress(1.0)
    status.text("Analysis complete!")
    st.session_state['stage'] = 'done'

# --- Screens ---
def show_upload_screen():
    st.title("🔬 AI Research Gap Analyzer")
    st.markdown("Upload at least **3 PDFs** of academic papers and provide your research topic. The system will cluster themes and surface genuine research gaps.")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    topic = st.text_input("Research topic", placeholder="e.g., deep learning for medical image segmentation")
    col1, col2 = st.columns(2)
    with col1:
        if uploaded_files:
            st.metric("PDFs selected", len(uploaded_files))
    with col2:
        if topic:
            st.success("Topic set")

    if st.button("Analyze Papers", type="primary", disabled=(not uploaded_files or not topic)):
        if len(uploaded_files) < 3:
            st.error("Please upload at least 3 PDFs.")
            return
        if len(uploaded_files) > 50:
            st.error("Maximum 50 PDFs allowed.")
            return
        st.session_state['uploaded_files'] = uploaded_files
        st.session_state['topic'] = topic
        st.session_state['stage'] = 1
        st.session_state['error'] = None
        st.session_state['failure_log'] = []
        st.rerun()

def show_processing_screen():
    st.title("Processing...")
    st.info("This may take a few minutes. Please wait.")
    try:
        run_full_pipeline()
    except Exception as e:
        st.session_state['error'] = str(e)
        st.session_state['stage'] = 0
    st.rerun()

def show_results_screen():
    st.title("📊 Research Gap Analysis Results")
    gap_report = st.session_state.get('gap_report')
    summaries = st.session_state.get('summaries', [])
    papers = st.session_state.get('papers', [])
    topic = st.session_state.get('topic', "")
    failure_log = st.session_state.get('failure_log', [])

    if isinstance(gap_report, dict) and 'error' in gap_report:
        st.error(f"Gap analysis failed: {gap_report['error']}")
        if st.button("Retry Analysis"):
            st.session_state['stage'] = 0
            st.rerun()
        return

    n_papers = len([p for p in papers if p.get('readable')])
    n_themes = len(summaries)
    n_gaps = len(gap_report.get('gaps', []))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers Analysed", n_papers)
    with col2:
        st.metric("Themes Found", n_themes)
    with col3:
        st.metric("Gaps Identified", n_gaps)

    # --- Interactive Plotly chart ---
    st.subheader("📈 Coverage by Theme")
    if summaries:
        df = pd.DataFrame({
            'Theme': [f"Theme {i}" for i, s in enumerate(summaries)],
            'Chunks': [s['chunk_count'] for s in summaries],
            'Percentage': [s['percentage'] for s in summaries],
            'Sparse': [s.get('sparse', False) for s in summaries],
            'Summary': [s['summary'][:120] + '...' for s in summaries]
        })
        color_map = {True: '#FFA500', False: '#1f77b4'}
        fig = px.bar(df, x='Chunks', y='Theme', color='Sparse',
                     color_discrete_map=color_map,
                     title='Theme Coverage (orange = sparse/understudied)',
                     hover_data=['Percentage', 'Summary'],
                     orientation='h')
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No themes found.")

    # --- Failure / Diagnostics section ---
    if failure_log:
        with st.expander("⚠️ Diagnostics & Warnings"):
            for msg in failure_log:
                st.warning(msg)

    # --- Rich gap cards ---
    st.subheader("🔍 Identified Research Gaps")
    if not gap_report.get('gaps'):
        st.write("No gaps identified.")
    else:
        for idx, gap in enumerate(gap_report['gaps']):
            with st.expander(f"{gap['title']}", expanded=(idx==0)):
                priority_colors = {"High": "#dc3545", "Medium": "#fd7e14", "Exploratory": "#6c757d"}
                color = priority_colors.get(gap['priority'], "#000")
                st.markdown(f"""
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <span style="background-color:{color}; color:white; padding:3px 10px; border-radius:12px; font-weight:bold;">{gap['priority']}</span>
                    <span style="background-color:#e9ecef; color:#333; padding:3px 10px; border-radius:12px;">{gap['type']}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Description**")
                st.write(gap['description'])

                evidence_themes = gap.get('evidence_themes', [])
                if evidence_themes:
                    st.markdown("**Evidence from themes**")
                    for t_idx in evidence_themes:
                        if t_idx < len(summaries):
                            snippet = summaries[t_idx]['summary'][:200] + "..."
                            st.caption(f"Theme {t_idx}: {snippet}")

                st.markdown("**Suggested direction**")
                st.info(gap['suggested_direction'])

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Gap {idx+1} of {n_gaps}")
                with col2:
                    if evidence_themes:
                        max_theme_pct = max([summaries[t]['percentage'] for t in evidence_themes if t < len(summaries)])
                        st.caption(f"Max theme coverage: {max_theme_pct:.1f}%")

    # --- Export section ---
    st.subheader("📥 Export Report")
    col1, col2, col3 = st.columns(3)
    with col1:
        export_markdown(gap_report, summaries, papers, topic)
    with col2:
        export_pdf(gap_report, summaries, papers, topic)
    with col3:
        export_docx(gap_report, summaries, papers, topic)

    if st.button("Start New Analysis"):
        for key in ['papers', 'chunks', 'embeddings', 'clusters', 'summaries', 'gap_report', 'topic', 'stage', 'error', 'failure_log']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['stage'] = 0
        st.rerun()

def main():
    if st.session_state.get('error') and st.session_state.get('stage') == 0:
        st.error(f"An error occurred: {st.session_state.error}")
        st.session_state.error = None

    if st.session_state['stage'] == 0:
        show_upload_screen()
    elif st.session_state['stage'] in [1,2,3,4,5,6]:
        show_processing_screen()
    elif st.session_state['stage'] == 'done':
        show_results_screen()

if __name__ == "__main__":
    main()