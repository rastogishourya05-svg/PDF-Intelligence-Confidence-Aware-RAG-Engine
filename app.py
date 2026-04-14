"""
Chat with your PDF — Confidence-Aware RAG with Source Transparency
Streamlit UI with per-sentence grounding labels and faithfulness scoring
"""

import os
import time
import tempfile
import traceback

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Sora', sans-serif;
    background: #0d0f14;
    color: #e2e8f0;
}

.main .block-container { padding: 2rem 2rem 4rem; max-width: 1300px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12151d !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.sidebar-logo-text { font-size: 16px; font-weight: 700; color: #e2e8f0; }
.sidebar-logo-sub { font-size: 11px; color: #64748b; }

/* ── Header ── */
.page-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #151822 100%);
    border: 1px solid #252a3d;
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 250px; height: 250px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
}
.page-header h1 {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #e2e8f0 30%, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.page-header p { font-size: 0.95rem; color: #64748b; line-height: 1.6; }

.header-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1rem; }
.pill {
    font-size: 0.75rem; font-weight: 500; color: #818cf8;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px; padding: 4px 12px;
}

/* ── Upload Zone ── */
.upload-card {
    background: #12151d; border: 2px dashed #252a3d;
    border-radius: 16px; padding: 2.5rem;
    text-align: center; transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}
.upload-card:hover { border-color: #6366f1; }
.upload-icon { font-size: 3rem; margin-bottom: 1rem; }
.upload-title { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.upload-sub { font-size: 0.85rem; color: #475569; }

/* ── Processing Badge ── */
.processing-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8; border-radius: 20px; padding: 6px 14px;
    font-size: 0.8rem; font-weight: 500; margin-bottom: 1rem;
}

/* ── Confidence Banner ── */
.confidence-banner {
    border-radius: 12px; padding: 1rem 1.4rem;
    margin-bottom: 1.5rem; display: flex; align-items: flex-start; gap: 12px;
    border-left: 4px solid;
}
.conf-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.conf-body { flex: 1; }
.conf-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 2px; }
.conf-desc { font-size: 0.88rem; line-height: 1.5; opacity: 0.85; }

/* ── Faithfulness Meter ── */
.faithfulness-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 10px; margin-bottom: 1.5rem;
}
.faith-card {
    background: #12151d; border: 1px solid #1e2535;
    border-radius: 12px; padding: 1rem; text-align: center;
}
.faith-val { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.faith-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
.faith-bar-track { background: #1e2535; border-radius: 4px; height: 4px; margin-top: 8px; }
.faith-bar-fill { height: 4px; border-radius: 4px; }

/* ── Answer Block ── */
.answer-wrapper {
    background: #12151d; border: 1px solid #1e2535;
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem;
}
.answer-header {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #475569; margin-bottom: 1.2rem; font-weight: 600;
    display: flex; align-items: center; gap: 6px;
}

/* ── Sentence Attribution ── */
.sentence-line { display: inline; }

.sent-doc {
    background: rgba(16,185,129,0.08); border-bottom: 2px solid rgba(16,185,129,0.5);
    padding: 1px 0; border-radius: 2px; cursor: default;
    transition: background 0.2s;
}
.sent-doc:hover { background: rgba(16,185,129,0.18); }

.sent-inf {
    background: rgba(245,158,11,0.08); border-bottom: 2px dashed rgba(245,158,11,0.5);
    padding: 1px 0; border-radius: 2px; cursor: default;
    transition: background 0.2s;
}
.sent-inf:hover { background: rgba(245,158,11,0.18); }

.sent-unknown { color: #94a3b8; }

.sent-badge {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.08em; border-radius: 3px; padding: 1px 5px;
    vertical-align: middle; margin-left: 3px; font-family: 'JetBrains Mono', monospace;
}
.badge-doc { background: rgba(16,185,129,0.2); color: #10b981; }
.badge-inf { background: rgba(245,158,11,0.2); color: #f59e0b; }

/* ── Legend ── */
.legend-bar {
    display: flex; gap: 16px; flex-wrap: wrap;
    background: #0d0f14; border: 1px solid #1e2535;
    border-radius: 10px; padding: 10px 14px;
    margin-bottom: 1.5rem; font-size: 0.8rem; align-items: center;
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

/* ── Sources ── */
.sources-section { margin-top: 1.5rem; }
.source-card {
    background: #0d0f14; border: 1px solid #1e2535;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 10px;
    transition: border-color 0.2s;
}
.source-card:hover { border-color: #6366f1; }
.source-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
}
.source-page { font-size: 0.75rem; color: #818cf8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.source-sim { font-size: 0.72rem; color: #475569; }
.source-text { font-size: 0.82rem; color: #64748b; line-height: 1.6; }
.sim-bar { height: 3px; border-radius: 2px; background: #1e2535; margin-top: 8px; }
.sim-fill { height: 3px; border-radius: 2px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

/* ── Gaps Warning ── */
.gaps-card {
    background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.5rem;
}
.gaps-title { font-size: 0.8rem; font-weight: 600; color: #ef4444; margin-bottom: 6px; }
.gap-item { font-size: 0.82rem; color: #94a3b8; padding: 2px 0; }

/* ── Chat History ── */
.chat-item {
    background: #12151d; border: 1px solid #1e2535;
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 10px; cursor: pointer; transition: border-color 0.2s;
}
.chat-item:hover { border-color: #6366f1; }
.chat-q { font-size: 0.85rem; color: #e2e8f0; font-weight: 500; margin-bottom: 4px; }
.chat-conf { font-size: 0.75rem; }

/* ── Input Area ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #12151d !important;
    border: 1px solid #252a3d !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e2535 !important;
    color: #94a3b8 !important;
}

/* ── Progress / Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Metrics & other Streamlit elements ── */
[data-testid="metric-container"] {
    background: #12151d; border: 1px solid #1e2535; border-radius: 12px; padding: 1rem;
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

.stAlert { border-radius: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #252a3d; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Dividers ── */
hr { border-color: #1e2535 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #12151d !important;
    border: 2px dashed #252a3d !important;
    border-radius: 16px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #12151d !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
    border: 1px solid #1e2535 !important;
}

/* ── Code blocks ── */
code { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #475569; font-weight: 600; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
for key, default in {
    "vectorstore": None,
    "chunks": None,
    "pdf_name": None,
    "pdf_fingerprint": None,
    "embeddings": None,
    "chat_history": [],
    "current_result": None,
    "num_pages": 0,
    "num_chunks": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# Lazy imports (heavy, cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from rag_engine import get_embeddings
    return get_embeddings()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📄</div>
        <div>
            <div class="sidebar-logo-text">PDF Intelligence</div>
            <div class="sidebar-logo-sub">Confidence-Aware RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload any PDF to start asking questions",
        label_visibility="collapsed",
    )

    if uploaded_file:
        from rag_engine import pdf_fingerprint, load_and_chunk_pdf, build_vectorstore

        fp = pdf_fingerprint(uploaded_file.getvalue())

        if fp != st.session_state.pdf_fingerprint:
            with st.spinner("🔍 Analyzing document…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    embeddings = load_embeddings()
                    chunks, pages = load_and_chunk_pdf(tmp_path)
                    vectorstore = build_vectorstore(chunks, embeddings)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_fingerprint = fp
                    st.session_state.embeddings = embeddings
                    st.session_state.chat_history = []
                    st.session_state.current_result = None
                    st.session_state.num_pages = len(pages)
                    st.session_state.num_chunks = len(chunks)

                    os.unlink(tmp_path)
                    st.success("Document ready!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.pdf_name:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Document Info</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.num_pages)
        with col2:
            st.metric("Chunks", st.session_state.num_chunks)

        st.markdown(f"""
        <div style="background:#0d0f14; border:1px solid #1e2535; border-radius:10px; padding:10px 12px; margin-top:8px;">
            <div style="font-size:0.72rem; color:#475569; margin-bottom:3px;">ACTIVE DOCUMENT</div>
            <div style="font-size:0.82rem; color:#818cf8; font-weight:500; word-break:break-all;">{st.session_state.pdf_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ RAG Settings"):
        top_k = st.slider("Retrieved Chunks (k)", 3, 10, 5, help="More chunks = more context but slower")
        st.markdown('<div style="font-size:0.78rem; color:#475569; margin-top:6px;">Higher k improves coverage for complex questions.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Legend in sidebar
    st.markdown("""
    <div style="background:#0d0f14; border:1px solid #1e2535; border-radius:12px; padding:1rem;">
        <div class="section-label">Answer Attribution</div>
        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.9;">
            <span style="background:rgba(16,185,129,0.15); border-bottom:2px solid #10b981; padding:1px 4px; border-radius:2px;">Underlined green</span> = Grounded in document<br>
            <span style="background:rgba(245,158,11,0.15); border-bottom:2px dashed #f59e0b; padding:1px 4px; border-radius:2px;">Dashed amber</span> = Model inference
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_result = None
            st.rerun()

# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📄 Chat with your PDF</h1>
    <p>Upload any document and ask questions. Every answer is annotated — see exactly what comes from the document versus what the AI infers.</p>
    <div class="header-pills">
        <span class="pill">Two-Pass RAG</span>
        <span class="pill">Faithfulness Scoring</span>
        <span class="pill">Per-Sentence Attribution</span>
        <span class="pill">Source Transparency</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Query Input
# ─────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-icon">☝️</div>
        <div class="upload-title">Upload a PDF in the sidebar to get started</div>
        <div class="upload-sub">Any PDF works — research papers, contracts, manuals, reports</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Example use cases</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#12151d; border:1px solid #1e2535; border-radius:14px; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:8px;">⚖️</div>
            <div style="font-weight:600; font-size:0.9rem; color:#e2e8f0; margin-bottom:4px;">Legal Documents</div>
            <div style="font-size:0.8rem; color:#475569;">Know exactly which clause backs each answer. Critical for contract review.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#12151d; border:1px solid #1e2535; border-radius:14px; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:8px;">🔬</div>
            <div style="font-weight:600; font-size:0.9rem; color:#e2e8f0; margin-bottom:4px;">Research Papers</div>
            <div style="font-size:0.8rem; color:#475569;">Distinguish between paper findings and model extrapolation instantly.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#12151d; border:1px solid #1e2535; border-radius:14px; padding:1.2rem;">
            <div style="font-size:1.5rem; margin-bottom:8px;">💊</div>
            <div style="font-weight:600; font-size:0.9rem; color:#e2e8f0; margin-bottom:4px;">Medical Reports</div>
            <div style="font-size:0.8rem; color:#475569;">Safely explore patient documents with grounded, transparent answers.</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # ── Question Input ──
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the key findings? What does section 3 say about liability?",
            label_visibility="collapsed",
        )
        col_ask, col_example = st.columns([4, 1])
        with col_ask:
            submitted = st.form_submit_button("🔍 Analyze", use_container_width=True, type="primary")
        with col_example:
            st.form_submit_button("↩ Clear", use_container_width=True)

    # ── Process Query ──
    if submitted and question.strip():
        from rag_engine import query_pdf

        with st.spinner("Running two-pass analysis…"):
            progress_bar = st.progress(0)
            status = st.empty()

            status.markdown('<div class="processing-badge">🔎 Pass 1: Retrieving relevant chunks…</div>', unsafe_allow_html=True)
            progress_bar.progress(20)
            time.sleep(0.2)

            status.markdown('<div class="processing-badge">🧠 Scoring faithfulness…</div>', unsafe_allow_html=True)
            progress_bar.progress(50)

            try:
                result = query_pdf(
                    question=question.strip(),
                    vectorstore=st.session_state.vectorstore,
                    top_k=top_k if 'top_k' in dir() else 5,
                )
                progress_bar.progress(80)
                status.markdown('<div class="processing-badge">✍️ Generating calibrated answer…</div>', unsafe_allow_html=True)
                time.sleep(0.2)
                progress_bar.progress(100)
                status.empty()
                progress_bar.empty()

                st.session_state.current_result = result
                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "result": result,
                    "ts": time.time(),
                })

            except Exception as e:
                status.empty()
                progress_bar.empty()
                st.error(f"Error: {str(e)}")
                with st.expander("Debug"):
                    st.code(traceback.format_exc())

    # ── Display Result ──
    if st.session_state.current_result:
        result = st.session_state.current_result
        faith = result["faithfulness"]
        conf = result["confidence_level"]
        sentences = result["sentences"]
        sources = result["sources"]

        st.markdown("---")

        # Confidence Banner
        banner_style = f"background:{conf['bg']}22; border-left-color:{conf['color']}; color:{conf['color']};"
        st.markdown(f"""
        <div class="confidence-banner" style="{banner_style}">
            <div class="conf-icon">{conf['icon']}</div>
            <div class="conf-body">
                <div class="conf-label" style="color:{conf['color']};">{conf['label']}</div>
                <div class="conf-desc" style="color:{conf['color']}99;">{conf['description']}</div>
                <div style="font-size:0.78rem; margin-top:4px; opacity:0.7;">{faith.get('assessment', '')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Faithfulness Scores
        rel_score = faith.get("relevance_score", 0)
        cov_score = faith.get("coverage_score", 0)
        ov_score = faith.get("overall_faithfulness", 0)

        def score_color(v):
            if v >= 0.75: return "#10b981"
            if v >= 0.5: return "#f59e0b"
            return "#ef4444"

        st.markdown(f"""
        <div class="faithfulness-row">
            <div class="faith-card">
                <div class="faith-val" style="color:{score_color(rel_score)}">{rel_score:.0%}</div>
                <div class="faith-lbl">Relevance</div>
                <div class="faith-bar-track"><div class="faith-bar-fill" style="width:{rel_score*100:.0f}%; background:{score_color(rel_score)};"></div></div>
            </div>
            <div class="faith-card">
                <div class="faith-val" style="color:{score_color(cov_score)}">{cov_score:.0%}</div>
                <div class="faith-lbl">Coverage</div>
                <div class="faith-bar-track"><div class="faith-bar-fill" style="width:{cov_score*100:.0f}%; background:{score_color(cov_score)};"></div></div>
            </div>
            <div class="faith-card">
                <div class="faith-val" style="color:{score_color(ov_score)}">{ov_score:.0%}</div>
                <div class="faith-lbl">Faithfulness</div>
                <div class="faith-bar-track"><div class="faith-bar-fill" style="width:{ov_score*100:.0f}%; background:{score_color(ov_score)};"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Knowledge Gaps Warning
        gaps = faith.get("gaps", [])
        if gaps:
            gap_items = "".join([f'<div class="gap-item">• {g}</div>' for g in gaps])
            st.markdown(f"""
            <div class="gaps-card">
                <div class="gaps-title">⚠️ Document gaps — AI may be inferring these:</div>
                {gap_items}
            </div>
            """, unsafe_allow_html=True)

        # Annotated Answer
        st.markdown('<div class="section-label">Annotated Answer</div>', unsafe_allow_html=True)

        # Build annotated HTML
        annotated_html = '<div class="answer-wrapper">'
        annotated_html += '<div class="answer-header">📝 Answer with Source Attribution</div>'
        annotated_html += '<div style="font-size:0.92rem; line-height:2.2; color:#cbd5e1;">'

        for sent in sentences:
            clean = sent["clean_text"].replace("<", "&lt;").replace(">", "&gt;")
            label = sent["label"]
            if label == "DOC":
                annotated_html += (
                    f'<span class="sent-doc" title="Grounded in document">{clean}'
                    f'<span class="sent-badge badge-doc">DOC</span></span> '
                )
            elif label == "INF":
                annotated_html += (
                    f'<span class="sent-inf" title="Model inference — not directly in document">{clean}'
                    f'<span class="sent-badge badge-inf">INF</span></span> '
                )
            else:
                annotated_html += f'<span class="sent-unknown">{clean}</span> '

        annotated_html += '</div></div>'
        st.markdown(annotated_html, unsafe_allow_html=True)

        # Attribution stats
        doc_count = sum(1 for s in sentences if s["label"] == "DOC")
        inf_count = sum(1 for s in sentences if s["label"] == "INF")
        total = len(sentences)

        if total > 0:
            doc_pct = doc_count / total * 100
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-bottom:1.5rem; flex-wrap:wrap;">
                <div style="font-size:0.8rem; color:#10b981;">
                    ✅ {doc_count} sentence{'s' if doc_count != 1 else ''} ({doc_pct:.0f}%) grounded in document
                </div>
                <div style="font-size:0.8rem; color:#f59e0b;">
                    ⚠️ {inf_count} sentence{'s' if inf_count != 1 else ''} ({100-doc_pct:.0f}%) model inference
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Source Chunks
        if sources:
            with st.expander(f"📚 View Source Chunks ({len(sources)} retrieved)", expanded=False):
                for i, src in enumerate(sources):
                    sim_pct = src["similarity"] * 100
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-page">Page {src['page']} · Chunk #{src['chunk_id']}</span>
                            <span class="source-sim">Similarity: {src['similarity']:.3f}</span>
                        </div>
                        <div class="source-text">{src['preview']}…</div>
                        <div class="sim-bar"><div class="sim-fill" style="width:{min(sim_pct,100):.0f}%;"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Chat History ──
    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.markdown('<div class="section-label">Previous Questions</div>', unsafe_allow_html=True)

        for i, item in enumerate(reversed(st.session_state.chat_history[:-1])):
            conf = item["result"]["confidence_level"]
            faith_score = item["result"]["faithfulness"].get("overall_faithfulness", 0)
            if st.button(
                f"{conf['icon']} {item['question'][:80]}{'…' if len(item['question']) > 80 else ''}",
                key=f"hist_{i}",
                use_container_width=True,
            ):
                st.session_state.current_result = item["result"]
                st.rerun()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid #1e2535; margin-top:2rem;">
    <div style="font-size:0.78rem; color:#334155; font-family:'JetBrains Mono', monospace;">
        PDF Intelligence · Two-Pass RAG · Powered by Groq + LangChain + FAISS
    </div>
</div>
""", unsafe_allow_html=True)