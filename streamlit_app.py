"""
streamlit_app.py — AI Know A Guy | Resume Screening UI
─────────────────────────────────────────────────────────
Run with:
    pip install streamlit pdfplumber anthropic openai
    streamlit run streamlit_app.py

Free API key options (no credit card needed):
    Groq   → https://console.groq.com          (Llama 3 models, generous free tier)
    Gemini → https://aistudio.google.com/apikey (Google AI Studio, free)
"""

import math
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

#  Page config (must be first Streamlit call)
st.set_page_config(
    page_title="AI Know A Guy — Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import our pipeline components 
from filtering import (
    ResumeParser,
    EmbeddingEngine,
    LLMScorer,
    PipelineConfig,
    ResumeScore,
    RUBRIC,
    JOB_DESCRIPTION,
)

# Helpers

@st.cache_resource
def get_embedder() -> EmbeddingEngine:
    """Cache the embedding engine so it's only initialised once per session."""
    e = EmbeddingEngine()
    e.prime(JOB_DESCRIPTION)
    return e


def score_text(resume_text: str, provider: str, api_key: str) -> ResumeScore:
    """Run the full scoring pipeline on a raw text string."""
    config = PipelineConfig(llm_provider=provider)
    scorer = LLMScorer(config, api_key=api_key, provider=provider)
    embedder = get_embedder()

    embed_sim = embedder.similarity(resume_text)
    llm = scorer.score(resume_text)

    r = ResumeScore(filename="uploaded")
    r.name                  = llm.get("name", "Unknown")
    r.embedding_similarity  = embed_sim
    r.technical_skills      = float(llm.get("technical_skills", 0))
    r.ai_ml_depth           = float(llm.get("ai_ml_depth", 0))
    r.shipping_velocity     = float(llm.get("shipping_velocity", 0))
    r.systems_thinking      = float(llm.get("systems_thinking", 0))
    r.clarity_communication = float(llm.get("clarity_communication", 0))
    r.strengths             = llm.get("strengths", [])
    r.gaps                  = llm.get("gaps", [])
    r.one_line_summary      = llm.get("one_line_summary", "")
    r.llm_error             = not bool(llm)

    r.llm_weighted_score = (
        r.technical_skills      * RUBRIC["technical_skills"]["weight"]
        + r.ai_ml_depth         * RUBRIC["ai_ml_depth"]["weight"]
        + r.shipping_velocity   * RUBRIC["shipping_velocity"]["weight"]
        + r.systems_thinking    * RUBRIC["systems_thinking"]["weight"]
        + r.clarity_communication * RUBRIC["clarity_communication"]["weight"]
    )
    llm_norm = r.llm_weighted_score / 10.0
    r.final_score = round(0.70 * llm_norm + 0.30 * embed_sim, 4)
    return r


def score_color(score: float) -> str:
    """Return a hex color for a 0-1 score."""
    if score >= 0.75:
        return "#2ecc71"   # green
    elif score >= 0.55:
        return "#f39c12"   # amber
    else:
        return "#e74c3c"   # red


def relevance_label(score: float) -> str:
    if score >= 0.78:
        return "🟢 Excellent match — top 5% territory"
    elif score >= 0.68:
        return "🟡 Strong match — top 15%"
    elif score >= 0.55:
        return "🟠 Moderate match — worth a look"
    else:
        return "🔴 Weak match — significant gaps"


def dimension_bar(label: str, score: float, weight: float):
    """Render a single dimension as a labeled progress bar."""
    pct = score / 10.0
    color = score_color(pct)
    st.markdown(
        f"""
        <div style="margin-bottom:10px">
          <div style="display:flex; justify-content:space-between; margin-bottom:3px">
            <span style="font-size:0.85rem; font-weight:600">{label}</span>
            <span style="font-size:0.85rem; color:{color}; font-weight:700">{score:.0f}/10
              <span style="color:#888; font-weight:400; font-size:0.75rem">({int(weight*100)}% weight)</span>
            </span>
          </div>
          <div style="background:#e0e0e0; border-radius:6px; height:10px; overflow:hidden">
            <div style="width:{pct*100:.1f}%; background:{color}; height:10px; border-radius:6px;
                        transition:width 0.4s ease"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def big_score_card(final: float, name: str):
    """Render the big score hero card."""
    color = score_color(final)
    pct = int(final * 100)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 2px solid {color};
            border-radius: 16px;
            padding: 24px 32px;
            text-align: center;
            margin-bottom: 20px;
        ">
          <div style="font-size:0.9rem; color:#666; margin-bottom:4px">Candidate</div>
          <div style="font-size:1.5rem; font-weight:700; margin-bottom:12px">{name}</div>
          <div style="font-size:3.5rem; font-weight:900; color:{color}; line-height:1">{pct}</div>
          <div style="font-size:1rem; color:#888; margin-top:4px">out of 100</div>
          <div style="font-size:1rem; margin-top:12px; font-weight:600; color:{color}">
            {relevance_label(final)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar 
with st.sidebar:
    st.image("https://img.shields.io/badge/AI%20Know%20A%20Guy-Resume%20Screener-blue?style=for-the-badge")
    st.markdown("---")

    st.markdown("### 🔑 API Configuration")
    st.markdown(
        """
        **Free options (no credit card):**
        - 🟢 [Groq](https://console.groq.com) — fastest, completely free
        - 🟡 [Google AI Studio](https://aistudio.google.com/apikey) — Gemini, free
        - 🔵 [Anthropic](https://console.anthropic.com) — $5 free credits on signup
        - 🔵 [OpenAI](https://platform.openai.com) — pay-as-you-go
        """
    )

    provider = st.selectbox(
        "LLM Provider",
        options=["groq", "anthropic", "openai"],
        index=0,
        help="Groq is free and fast — recommended for testing",
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your key here...",
        help="Stored only in your browser session. Never sent anywhere except the chosen API.",
    )

    if not api_key:
        st.warning("⚠️ No API key — embedding-only scoring (less accurate).")
    else:
        st.success("✅ API key set")

    st.markdown("---")
    st.markdown("### ℹ️ How scoring works")
    st.markdown(
        """
        **Final Score = 70% LLM + 30% Semantic**

        | Dimension | Weight |
        |-----------|--------|
        | Technical Skills | 30% |
        | AI/ML Depth | 25% |
        | Shipping Velocity | 20% |
        | Systems Thinking | 15% |
        | Clarity | 10% |
        """
    )
    st.markdown("---")
    st.caption("Built for AI Know A Guy hiring challenge")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("AI Know A Guy — Resume Screener")
st.markdown(
    "Upload a resume to score it against the **Senior Software Engineer** role, "
    "or batch-screen up to 20 resumes at once."
)

tab_single, tab_batch, tab_jd = st.tabs(
    ["📄 Score One Resume", "📦 Batch Screen", "📋 View Job Description"]
)

# TAB 1 — Single Resume Scorer
with tab_single:
    st.markdown("### Upload your resume")
    st.markdown(
        "Upload a PDF (or paste plain text below). "
        "The system will score it across 5 dimensions and tell you "
        "how relevant it is to this role."
    )

    col_upload, col_paste = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload resume PDF",
            type=["pdf"],
            help="Max 10 MB. Text is extracted locally, then sent to your chosen LLM.",
        )
        resume_text_from_pdf = ""
        if uploaded_file:
            # Windows-safe temp file: write → explicitly close → parse → delete
            tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            try:
                with os.fdopen(tmp_fd, "wb") as tmp:
                    tmp.write(uploaded_file.read())
                tmp_path = Path(tmp_name)
                parser = ResumeParser()
                text, err = parser.parse(tmp_path)
            finally:
                try:
                    os.unlink(tmp_name)   # always clean up the temp file
                except OSError:
                    pass
            if err or len(text.strip()) < 50:
                st.error(" Could not extract text from this PDF. Try the paste box instead.")
            else:
                resume_text_from_pdf = text
                st.success(f" Extracted {len(text):,} characters from PDF")
                with st.expander("Preview extracted text"):
                    st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

    with col_paste:
        pasted_text = st.text_area(
            "Or paste resume text here",
            height=220,
            placeholder="Paste plain text of your resume here if PDF parsing fails...",
        )

    # Resolve which text to use
    final_resume_text = resume_text_from_pdf or pasted_text.strip()

    st.markdown("---")
    score_btn = st.button(" Score This Resume", type="primary", disabled=not final_resume_text)

    if score_btn and final_resume_text:
        with st.spinner("Analysing resume... (takes ~5 seconds)"):
            result = score_text(final_resume_text, provider, api_key)

        st.markdown("---")
        st.markdown("## Results")

        # ── Score card + dimension breakdown ──────────────────────────────────
        col_card, col_dims = st.columns([1, 1], gap="large")

        with col_card:
            big_score_card(result.final_score, result.name)

            # Score composition breakdown
            st.markdown("**Score composition:**")
            llm_contribution  = round(0.70 * (result.llm_weighted_score / 10) * 100, 1)
            emb_contribution  = round(0.30 * result.embedding_similarity * 100, 1)
            st.markdown(
                f"""
                | Signal | Raw | Contribution |
                |--------|-----|-------------|
                | LLM Rubric (70%) | {result.llm_weighted_score:.1f}/10 | {llm_contribution:.1f} pts |
                | Semantic Match (30%) | {result.embedding_similarity:.2f}/1.0 | {emb_contribution:.1f} pts |
                | **Final** | | **{int(result.final_score*100)} / 100** |
                """
            )

        with col_dims:
            st.markdown("**Dimension Scores**")
            dimension_bar(" Technical Skills",      result.technical_skills,      RUBRIC["technical_skills"]["weight"])
            dimension_bar(" AI / ML Depth",         result.ai_ml_depth,           RUBRIC["ai_ml_depth"]["weight"])
            dimension_bar(" Shipping Velocity",     result.shipping_velocity,     RUBRIC["shipping_velocity"]["weight"])
            dimension_bar("  Systems Thinking",     result.systems_thinking,      RUBRIC["systems_thinking"]["weight"])
            dimension_bar("  Clarity / Comms",      result.clarity_communication, RUBRIC["clarity_communication"]["weight"])

        # ── Summary + Strengths + Gaps ────────────────────────────────────────
        st.markdown("---")
        if result.one_line_summary:
            st.info(f" **AI Summary:** {result.one_line_summary}")

        col_str, col_gap = st.columns([1, 1], gap="large")
        with col_str:
            st.markdown("#### ✅ Strengths")
            if result.strengths:
                for s in result.strengths:
                    st.markdown(f"- {s}")
            else:
                st.markdown("_No strengths detected (LLM unavailable?)_")

        with col_gap:
            st.markdown("#### ⚠️ Gaps")
            if result.gaps:
                for g in result.gaps:
                    st.markdown(f"- {g}")
            else:
                st.markdown("_No gaps detected_")

        # ── Improvement tips ──────────────────────────────────────────────────
        if result.final_score < 0.72:
            st.markdown("---")
            st.markdown("#### 💡 How to improve your score for this role")
            tips = []
            if result.technical_skills < 7:
                tips.append("Add **specific versions** of tools you've used (e.g. PyTorch 2.x, LangChain 0.3)")
            if result.ai_ml_depth < 7:
                tips.append("Add **hard metrics** to every AI project (accuracy %, latency ms, cost reduction $)")
            if result.shipping_velocity < 7:
                tips.append("Mention **GitHub links**, deployment platforms (Fly.io, Railway, HuggingFace Spaces)")
            if result.systems_thinking < 7:
                tips.append("Describe **architecture decisions** you made and why (e.g. 'chose RAG over fine-tuning because...')")
            if result.clarity_communication < 7:
                tips.append("Start every bullet with a strong **action verb** and end with a **quantified result**")
            for tip in tips:
                st.markdown(f"- {tip}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Screen
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Batch Screen Multiple Resumes")
    st.markdown(
        "Upload up to **20 PDF resumes** at once. "
        "They'll be scored and ranked — top candidates highlighted in green."
    )

    batch_files = st.file_uploader(
        "Upload resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.markdown(f"**{len(batch_files)} file(s) uploaded.**")
        run_batch = st.button(" Run Batch Screening", type="primary")

        if run_batch:
            parser = ResumeParser()
            results: list[ResumeScore] = []

            progress_bar = st.progress(0, text="Starting...")
            status_area  = st.empty()

            for i, f in enumerate(batch_files):
                status_area.markdown(f"Scoring **{f.name}**…")
                # Windows-safe temp file pattern
                tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
                try:
                    with os.fdopen(tmp_fd, "wb") as tmp:
                        tmp.write(f.read())
                    tmp_path = Path(tmp_name)
                    text, err = parser.parse(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                if err or len(text.strip()) < 50:
                    r = ResumeScore(filename=f.name, parse_error=True,
                                    one_line_summary="Could not parse PDF")
                else:
                    r = score_text(text, provider, api_key)
                    r.filename = f.name
                results.append(r)

                progress_bar.progress((i + 1) / len(batch_files),
                                      text=f"Scored {i+1}/{len(batch_files)}")
                time.sleep(0.2)  # small delay between calls

            progress_bar.empty()
            status_area.empty()

            # Sort by final score
            results.sort(key=lambda r: r.final_score, reverse=True)
            for i, r in enumerate(results, 1):
                r.rank = i

            cutoff_score = 0.72   # approx top 5%
            top_n = max(1, math.ceil(len(results) * 0.05))

            st.success(f"✅ Done! Screened {len(results)} resumes. "
                       f"Top {top_n} highlighted below.")

            # ── Results table ─────────────────────────────────────────────────
            import pandas as pd

            rows = []
            for r in results:
                rows.append({
                    "Rank":        r.rank,
                    "Name":        r.name,
                    "File":        r.filename,
                    "Score /100":  int(r.final_score * 100),
                    "Relevance":   relevance_label(r.final_score).split("—")[0].strip(),
                    "Technical":   f"{r.technical_skills:.0f}/10",
                    "AI/ML":       f"{r.ai_ml_depth:.0f}/10",
                    "Shipping":    f"{r.shipping_velocity:.0f}/10",
                    "Systems":     f"{r.systems_thinking:.0f}/10",
                    "Summary":     r.one_line_summary[:80],
                })

            df = pd.DataFrame(rows)

            # Highlight top candidates
            def highlight_row(row):
                score = int(row["Score /100"])
                if score >= 75:
                    return ["background-color: #d4edda"] * len(row)
                elif score >= 60:
                    return ["background-color: #fff3cd"] * len(row)
                else:
                    return [""] * len(row)

            st.dataframe(
                df.style.apply(highlight_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # ── Expandable detail cards for top 5 ────────────────────────────
            st.markdown("---")
            st.markdown("### 🏆 Top Candidates — Detailed View")
            for r in results[:min(5, len(results))]:
                with st.expander(
                    f"#{r.rank} — {r.name}  |  Score: {int(r.final_score*100)}/100  |  "
                    f"{relevance_label(r.final_score)}",
                    expanded=(r.rank == 1),
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        dimension_bar("Technical Skills",  r.technical_skills,      0.30)
                        dimension_bar("AI / ML Depth",     r.ai_ml_depth,           0.25)
                        dimension_bar("Shipping Velocity", r.shipping_velocity,     0.20)
                        dimension_bar("Systems Thinking",  r.systems_thinking,      0.15)
                        dimension_bar("Clarity / Comms",   r.clarity_communication, 0.10)
                    with c2:
                        if r.one_line_summary:
                            st.info(f"💬 {r.one_line_summary}")
                        st.markdown("**✅ Strengths**")
                        for s in r.strengths:
                            st.markdown(f"- {s}")
                        st.markdown("**⚠️ Gaps**")
                        for g in r.gaps:
                            st.markdown(f"- {g}")

            # ── Download CSV ──────────────────────────────────────────────────
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_data,
                file_name="screening_results.csv",
                mime="text/csv",
            )

# TAB 3 — Job Description
with tab_jd:
    st.markdown("### Job Description — Senior Software Engineer @ AI Know A Guy")
    st.markdown(
        "_This is the target the scoring system is calibrated against. "
        "Every resume is compared to this._"
    )
    st.markdown("---")
    st.markdown(JOB_DESCRIPTION)

    st.markdown("---")
    st.markdown("### Scoring Rubric Weights")
    for dim, meta in RUBRIC.items():
        st.markdown(f"**{dim.replace('_', ' ').title()}** ({int(meta['weight']*100)}%) — {meta['description']}")
