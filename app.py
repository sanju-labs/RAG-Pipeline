"""
Multimodal RAG Chat — "Hi, I'm Bing!"
"""

import os
import streamlit as st
import config as cfg
from rag_engine import MultimodalRAG

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(page_title="Bing", page_icon="🔮", layout="wide", initial_sidebar_state="auto")

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset ── */
.stApp { background: #09090f; font-family: 'Inter', sans-serif; }
header, .stDeployButton, #MainMenu, footer { display: none !important; }
[data-testid="stBottomBlockContainer"] { background: #09090f !important; }

/* ── Chat Input — floating pill ── */
[data-testid="stChatInput"] { max-width: 720px; margin: 0 auto; }
[data-testid="stChatInput"] textarea {
    background: #131320 !important; border: 1px solid #23233a !important;
    border-radius: 26px !important; color: #e4e4ec !important;
    padding: 14px 20px !important; font-size: 14.5px !important;
    transition: border-color 0.2s;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a4a66 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 24px rgba(108,99,255,0.1) !important;
}
[data-testid="stChatInput"] button {
    background: #6c63ff !important; border-radius: 50% !important;
    color: #fff !important; border: none !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 10px 0 !important; max-width: 720px; margin: 0 auto;
}
[data-testid="stChatMessage"] p { color: #cdcdd8 !important; font-size: 14.5px; line-height: 1.7; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0b0b14 !important; border-right: 1px solid #181828 !important; }
section[data-testid="stSidebar"] * { color: #b0b0c4 !important; font-size: 13.5px; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] strong { color: #e4e4ec !important; }
section[data-testid="stSidebar"] hr { border-color: #1a1a2a !important; margin: 12px 0 !important; }
section[data-testid="stSidebar"] code {
    background: #16162a !important; color: #a78bfa !important;
    padding: 1px 6px; border-radius: 4px; font-size: 12px;
}

/* ── Upload ── */
[data-testid="stFileUploader"] { border-radius: 12px !important; }
[data-testid="stFileUploader"] section { background: #0f0f1c !important; border: 1px dashed #23233a !important; border-radius: 12px !important; padding: 16px !important; }
[data-testid="stFileUploader"] button { background: #6c63ff !important; color: #fff !important; border-radius: 8px !important; font-size: 12px !important; }

/* ── Toggle ── */
[data-testid="stToggle"] label span { color: #a78bfa !important; font-weight: 500; font-size: 13px !important; }

/* ── BTS ── */
.bts-box {
    background: #0f0f1c; border: 1px solid #1c1c30; border-radius: 10px;
    padding: 13px 15px; margin: 7px 0; font-size: 12.5px; line-height: 1.65; color: #b0b0c4;
}
.bts-box strong { color: #c4b5fd !important; }
.bts-tag {
    display: inline-block; background: rgba(108,99,255,0.12); color: #a78bfa;
    padding: 2px 9px; border-radius: 14px; font-size: 10px;
    font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;
}

/* ── Title ── */
.hero { text-align: center; padding: 80px 20px 20px; }
.hero h1 {
    font-size: 2.8rem; font-weight: 600; margin: 0;
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #3d3d56; font-size: 13.5px; margin-top: 8px; font-weight: 400; }

/* ── Misc ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #23233a; border-radius: 4px; }
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stTextInput"] input {
    background: #0f0f1c !important; border: 1px solid #23233a !important;
    border-radius: 8px !important; color: #e4e4ec !important; font-size: 13px !important;
}
.stSpinner > div { border-top-color: #6c63ff !important; }
.stAlert { background: #0f0f1c !important; border: 1px solid #1c1c30 !important; border-radius: 10px !important; }

/* ── Sidebar toggle — restyle Streamlit's native button ── */
[data-testid="collapsedControl"] {
    position: fixed !important; top: 50% !important; left: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 999999 !important;
    background: #1a1a2e !important; border: 1px solid #23233a !important;
    border-left: none !important; border-radius: 0 8px 8px 0 !important;
    width: 22px !important; height: 56px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    margin: 0 !important; padding: 0 !important;
}
[data-testid="collapsedControl"]:hover { background: #23233a !important; }
[data-testid="collapsedControl"] svg { display: none !important; }
[data-testid="collapsedControl"]::after {
    content: '›'; color: #a78bfa; font-size: 16px; line-height: 1;
}
[data-testid="stSidebarCollapseButton"] button {
    position: fixed !important; top: 50% !important; left: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 999999 !important;
    background: #1a1a2e !important; border: 1px solid #23233a !important;
    border-left: none !important; border-radius: 0 8px 8px 0 !important;
    width: 22px !important; height: 56px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    margin: 0 !important; padding: 0 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover { background: #23233a !important; }
[data-testid="stSidebarCollapseButton"] button svg { display: none !important; }
[data-testid="stSidebarCollapseButton"] button::after {
    content: '‹'; color: #a78bfa; font-size: 16px; line-height: 1;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ── Session Init ─────────────────────────────────────────────────────

defaults = {"messages": [], "rag": None, "ingested": [], "last_bts": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.rag is None:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            st.session_state.rag = MultimodalRAG(api_key)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("### ⚙️ Setup")
    if st.session_state.rag:
        st.markdown("---")
        st.markdown("### 📁 Drop your files")
        files = st.file_uploader(
            "Drag & drop here", type=cfg.SUPPORTED_TYPES,
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if files:
            for f in files:
                already = [i["file"] for i in st.session_state.ingested]
                if f.name not in already:
                    with st.spinner(f"Processing {f.name}..."):
                        try:
                            result = st.session_state.rag.ingest(f.name, f.read())
                            if "error" not in result:
                                st.session_state.ingested.append(result)
                                st.success(f"✓ {f.name} — {result['chunks']} chunks")
                            else:
                                st.warning(result["error"])
                        except Exception as e:
                            st.error(f"Error processing {f.name}: {e}")
        if st.session_state.ingested:
            stats = st.session_state.rag.get_stats()
            st.caption(f"📊 {stats['doc_chunks']} chunks · {stats['cached_queries']} cached")

        st.markdown("---")
        bts_on = st.toggle("🚀 Curious about BTS? Click to explore!")

        if bts_on and st.session_state.last_bts:
            b = st.session_state.last_bts
            st.markdown("#### 🔍 Behind The Scenes")

            st.markdown(f"""<div class="bts-box">
            <span class="bts-tag">STEP 1 · YOUR QUERY</span><br>
            <strong>"{b['query']}"</strong><br>
            This is your raw question. The system will now find the best matching
            pieces from your documents to answer it.
            </div>""", unsafe_allow_html=True)

            if b.get("cache_hit"):
                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 2 · CACHE HIT ⚡</span><br>
                A nearly identical question was asked before!<br>
                <strong>Match: {b.get('cache_similarity', 0) * 100:.1f}%</strong>
                (threshold: {cfg.CACHE_THRESHOLD * 100:.0f}%)<br>
                The cached answer was returned instantly — <strong>zero LLM cost, zero search time</strong>.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 2 · EMBEDDING YOUR QUERY</span><br>
                Your question was converted into a <strong>{b['embedding_dim']}-dimension vector</strong>
                using <code>{b['embedding_model']}</code>.<br>
                This turns words into numbers so the system can measure
                how "close" your question is to each stored chunk.
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 3 · CACHE CHECK</span><br>
                Searched the semantic cache — <strong>no similar past query found</strong>.
                Proceeding to full hybrid search.
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 4 · HYBRID SEARCH</span><br>
                Searched <strong>{b.get('total_chunks_in_db', '?')}</strong> total chunks using two methods:<br><br>
                <strong>① Semantic Search</strong> (weight: {b['semantic_weight']}) —
                compares your query vector against every chunk vector using cosine distance.
                Found <strong>{b.get('semantic_candidates', '?')}</strong> candidates.<br><br>
                <strong>② BM25 Keyword Search</strong> (weight: {b['bm25_weight']}) —
                scores chunks by exact word overlap, boosted by rarity (TF-IDF style).
                Found <strong>{b.get('bm25_candidates', '?')}</strong> candidates.<br><br>
                Both sets are merged. Each chunk gets a fused score:<br>
                <code>score = {b['semantic_weight']}×semantic + {b['bm25_weight']}×bm25</code>
                </div>""", unsafe_allow_html=True)

                scores_html = ""
                for i, s in enumerate(b.get("scores", [])):
                    preview = b.get("chunk_previews", [""])[i][:55] if i < len(b.get("chunk_previews", [])) else ""
                    scores_html += (
                        f"<strong>#{i+1}</strong> · "
                        f"Combined: {s['combined'] * 100:.1f}% "
                        f"(sem: {s['semantic'] * 100:.0f}%, bm25: {s['bm25'] * 100:.0f}%)<br>"
                        f"<em style='color:#6b6b88'>  \"{preview}...\"</em><br>"
                    )
                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 5 · TOP-{b['top_k']} RESULTS</span><br>
                {scores_html}<br>
                Sources: <strong>{', '.join(b.get('sources', ['?']))}</strong>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="bts-box">
                <span class="bts-tag">STEP 6 · LLM GENERATION</span><br>
                The top <strong>{b.get('num_retrieved', '?')}</strong> chunks
                (<strong>{b.get('context_chars', 0):,}</strong> chars) were sent as context
                to <code>{b['llm_model']}</code>.<br><br>
                The LLM doesn't search — it reads. Its job is to
                synthesize the retrieved chunks into one clear answer.<br><br>
                Output: <strong>{b.get('answer_chars', 0):,}</strong> characters.
                </div>""", unsafe_allow_html=True)

        elif bts_on:
            st.caption("Ask a question first — the breakdown will appear here.")

    else:
        st.info("RAG engine not initialized. Check your OPENAI_API_KEY in .env.")

# ── Main Chat ─────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""<div class="hero">
        <h1>Hi, I'm Bing!</h1>
        <p>Drop your documents in the sidebar, then ask me anything</p>
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

if question := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    if not st.session_state.rag:
        answer = "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
    elif not st.session_state.ingested:
        answer = "Upload at least one document in the sidebar first."
    else:
        with st.spinner(""):
            try:
                result = st.session_state.rag.query(question)
                answer = result["answer"]
                st.session_state.last_bts = result.get("bts")
                if result["bts"].get("cache_hit"):
                    answer += "\n\n*⚡ Answered from cache*"
            except Exception as e:
                answer = f"Something went wrong: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
    st.rerun()