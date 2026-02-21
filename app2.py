import os
import math
import validators
import streamlit as st
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ujjwalize — Web Summarizer",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    :root {
        --bg:        #0f0f13;
        --surface:   #17171f;
        --surface2:  #1e1e28;
        --border:    rgba(255,255,255,0.07);
        --accent:    #e8714a;
        --accent2:   #f0a282;
        --text:      #ececec;
        --muted:     #7a7a8c;
        --radius:    14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2.5rem 3rem 4rem !important; max-width: 900px !important; }

    .hero {
        text-align: center;
        padding: 3.5rem 1rem 2rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(232,113,74,0.12);
        border: 1px solid rgba(232,113,74,0.3);
        color: var(--accent2);
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 1rem;
        letter-spacing: -0.02em;
        color: #fff;
    }
    .hero h1 span { color: var(--accent); }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        font-weight: 300;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .snap-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.6rem;
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color .2s !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(232,113,74,0.15) !important;
        outline: none !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        padding: 0.65rem 2rem !important;
        transition: opacity .18s, transform .18s !important;
        width: 100% !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    .stDownloadButton > button {
        background: var(--surface2) !important;
        color: var(--accent2) !important;
        border: 1px solid rgba(232,113,74,0.3) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        transition: background .18s !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(232,113,74,0.1) !important;
    }

    .stSlider > div { padding: 0.2rem 0 !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background: var(--surface2) !important;
        border-radius: 99px !important;
    }

    .stAlert {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background: var(--surface2) !important;
    }

    .result-box {
        background: var(--surface);
        border: 1px solid rgba(232,113,74,0.25);
        border-radius: var(--radius);
        padding: 2rem 2.2rem;
        margin-top: 1.5rem;
        line-height: 1.8;
        font-size: 1rem;
        color: var(--text);
        position: relative;
    }
    .result-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        border-radius: var(--radius) var(--radius) 0 0;
    }

    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        font-size: 0.8rem;
        color: var(--muted);
    }
    .stat-pill strong { color: var(--accent2); }

    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }
    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #fff;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-logo span { color: var(--accent); }
    .sidebar-section {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.5rem 0 0.5rem;
    }

    .stSpinner > div { border-top-color: var(--accent) !important; }

    [data-baseweb="select"] div {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    [data-baseweb="popover"] { background: var(--surface2) !important; }

    .tip-text {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌟 <span>Ujjwal</span>ize</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🔑 API Key</div>', unsafe_allow_html=True)

    _env_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    if not _env_key:
        _env_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        label_visibility="collapsed",
        placeholder="gsk_••••••••••••••••••••••",
    )

    if not groq_api_key.strip():
        groq_api_key = _env_key

    st.markdown('<div class="sidebar-section">🌐 Output Language</div>', unsafe_allow_html=True)
    output_language = st.selectbox("Language", ["English", "Hindi"], index=0, label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">📝 Summary Length</div>', unsafe_allow_html=True)
    summary_words = st.slider("Words", 80, 500, 300, 10, label_visibility="collapsed")
    st.caption(f"Target: **{summary_words} words**")

    st.markdown('<div class="sidebar-section">⚙️ Chunking</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size (chars)", 1500, 9000, 4500, 500)
    chunk_overlap = st.slider("Overlap (chars)", 0, 1000, 300, 50)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#7a7a8c;line-height:1.5;">'
        "Powered by <strong style='color:#e8714a'>Groq</strong> + "
        "<strong style='color:#e8714a'>LLaMA 3.1</strong>.<br>"
        "Built by <strong style='color:#e8714a'>Ujjwal</strong> 🌟</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🌐 AI-Powered Web Summarizer</div>
        <h1>Summarize any<br><span>website instantly</span></h1>
        <p>Paste any webpage URL and get a sharp, concise summary in seconds — powered by LLaMA 3.1.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DEMO URLS
# ─────────────────────────────────────────────
DEMO_URLS = {
    "— Choose a demo URL —": "",
    "🌐  GeeksForGeeks – Artificial Intelligence": "https://www.geeksforgeeks.org/artificial-intelligence/",
    "🌐  Wikipedia – Artificial Intelligence (EN)": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "🌐  Wikipedia – AI (Hindi)": "https://hi.wikipedia.org/wiki/कृत्रिम_बुद्धिमत्ता",
    "📰  BBC News – Technology": "https://www.bbc.com/news/technology",
    "📰  TechCrunch – Latest News": "https://techcrunch.com/",
}

st.markdown('<div class="snap-label">Try a demo</div>', unsafe_allow_html=True)
demo_choice = st.selectbox("Demo", list(DEMO_URLS.keys()), label_visibility="collapsed")

st.markdown('<div class="snap-label" style="margin-top:1rem;">Paste your website URL</div>', unsafe_allow_html=True)
url = st.text_input(
    "URL",
    value=DEMO_URLS.get(demo_choice, ""),
    placeholder="https://example.com/article",
    label_visibility="collapsed",
)

col_btn, col_tip = st.columns([1, 2])
with col_btn:
    run_btn = st.button("🌟 Summarize", type="primary")
with col_tip:
    st.markdown(
        '<p class="tip-text">Works best with Wikipedia, news articles, and blogs. '
        "JS-heavy pages may extract less content.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        overlap = max(0, size // 4)
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        ch = text[i:j].strip()
        if ch:
            chunks.append(ch)
        i = j - overlap
        if i < 0:
            i = 0
        if j == n:
            break
    return chunks


def load_website_text(page_url: str) -> str:
    r = requests.get(
        page_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
        timeout=25,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main = (
        soup.select_one("#mw-content-text")
        or soup.select_one("article")
        or soup.body
    )
    text = (
        main.get_text(separator="\n", strip=True)
        if main
        else soup.get_text(separator="\n", strip=True)
    )
    # Trim to 20k chars — enough for a great summary, avoids excessive chunks
    return text[:20000]


def build_llm(groq_key: str) -> ChatGroq:
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0)


def summarize_chunks(llm, chunks: list[str], language: str, target_words: int) -> str:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = StrOutputParser()

    # ── Fast path: single chunk fits in one call ──
    if len(chunks) == 1:
        stuff_prompt = PromptTemplate(
            input_variables=["text", "language", "words"],
            template=(
                "Summarize the text below in about {words} words in {language}.\n"
                "Rules: Do NOT add extra facts. Be clear, concise, and natural.\n\n"
                "TEXT:\n{text}\n"
            ),
        )
        progress_bar = st.progress(50, text="Summarizing…")
        result = (stuff_prompt | llm | parser).invoke(
            {"text": chunks[0], "language": language, "words": target_words}
        ).strip()
        progress_bar.progress(100, text="Done!")
        return result

    # ── Multi-chunk path: parallel map → reduce ──
    map_prompt = PromptTemplate(
        input_variables=["text", "language", "words"],
        template=(
            "Summarize the text below in {words} words in {language}.\n"
            "Rules: Do NOT add extra facts. Keep it clear and concise.\n\n"
            "TEXT:\n{text}\n"
        ),
    )
    reduce_prompt = PromptTemplate(
        input_variables=["summaries", "language", "words"],
        template=(
            "Combine the following chunk summaries into ONE final coherent summary "
            "of about {words} words in {language}.\n"
            "Rules:\n"
            "- Output ONLY in {language}. Do not mix languages.\n"
            "- Do NOT add extra facts.\n"
            "- Make it flow naturally as one piece.\n\n"
            "CHUNK SUMMARIES:\n{summaries}\n"
        ),
    )

    per_chunk_words = max(60, min(160, math.ceil(target_words / max(1, len(chunks)))))
    map_chain = map_prompt | llm | parser
    reduce_chain = reduce_prompt | llm | parser

    partials = [None] * len(chunks)
    progress_bar = st.progress(0, text=f"Processing {len(chunks)} chunks in parallel…")
    completed = 0

    def process_chunk(idx, ch):
        return idx, map_chain.invoke({"text": ch, "language": language, "words": per_chunk_words})

    # Run all map calls in parallel (Groq is fast, I/O bound — threads help a lot)
    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as executor:
        futures = {executor.submit(process_chunk, i, ch): i for i, ch in enumerate(chunks)}
        for future in as_completed(futures):
            idx, partial = future.result()
            partials[idx] = partial.strip()
            completed += 1
            pct = int(completed / len(chunks) * 90)
            progress_bar.progress(pct, text=f"Processed {completed} of {len(chunks)} chunks…")

    progress_bar.progress(95, text="Finalizing summary…")
    summaries_blob = "\n\n".join([f"• {p}" for p in partials if p])
    result = reduce_chain.invoke({
        "summaries": summaries_blob,
        "language": language,
        "words": target_words,
    }).strip()
    progress_bar.progress(100, text="Done!")
    return result


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn:
    if not groq_api_key.strip():
        st.error("🔑 Please enter your Groq API key in the sidebar.")
        st.stop()
    if not url.strip():
        st.error("🔗 Please paste a URL to summarize.")
        st.stop()
    if not validators.url(url):
        st.error("❌ That doesn't look like a valid URL. Double-check and try again.")
        st.stop()

    # Block YouTube URLs
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        st.error("⚠️ This app summarizes websites only. For YouTube videos, use the full Ujjwalize app.")
        st.stop()

    out_lang = "Hindi" if output_language == "Hindi" else "English"

    try:
        llm = build_llm(groq_api_key)

        with st.spinner("🔍 Fetching website content…"):
            text = load_website_text(url)

        if not text or len(text.strip()) < 120:
            st.error("⚠️ Not enough readable text found. Try a different URL.")
            st.stop()

        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            st.error("⚠️ Chunking failed — content may be empty.")
            st.stop()

        with st.spinner("✨ Summarizing…"):
            final_summary = summarize_chunks(llm, chunks, language=out_lang, target_words=summary_words)

        # ── Result Stats ──
        word_count = len(final_summary.split())
        char_count = len(text)

        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-pill">📄 Source: <strong>Website</strong></div>
                <div class="stat-pill">🔢 Chunks: <strong>{len(chunks)}</strong></div>
                <div class="stat-pill">📏 Input: <strong>{char_count:,} chars</strong></div>
                <div class="stat-pill">📝 Output: <strong>~{word_count} words</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Summary Box ──
        st.markdown(
            f'<div class="result-box">{final_summary}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Download Summary",
            data=final_summary,
            file_name="ujjwalize_web_summary.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error("Something went wrong. Try a different URL or check your API key.")
        with st.expander("🐛 Error details"):
            st.exception(e)
