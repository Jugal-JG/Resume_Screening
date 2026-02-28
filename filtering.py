"""
Architecture:
    1. PDF Parsing      →  raw text extraction (pdfplumber → PyMuPDF → skip)
    2. Embedding Rank   →  cosine similarity to job description
                           (TF-IDF built-in; upgrades to OpenAI ada-002 if key set)
    3. LLM Scoring      →  multi-dimensional rubric via Claude Haiku / GPT-4o-mini
    4. Score Fusion     →  70% LLM  +  30% embedding similarity
    5. Output           →  ranked CSV  +  detailed JSON  +  human-readable report

Quick start:
    pip install pdfplumber anthropic openai numpy
    export ANTHROPIC_API_KEY=sk-ant-...       # or OPENAI_API_KEY
    python filtering.py --resumes ./resumes --top 0.05

    # Demo mode (no resumes directory needed):
    python filtering.py --demo

48-hour ship checklist is printed at the bottom of the screening report.
"""

import os
import re
import csv
import json
import time
import math
# import argparse  # Streamlit-only mode
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Optional heavy deps (graceful degradation) 
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── Job Description ───────────────────────────────────────────────────────────
JOB_DESCRIPTION = """
Role: Senior Software Engineer
Company: AI Know A Guy

We are shipping a real AI product in 30 days and need someone who:
- Has strong Python and AI/ML engineering skills
- Has hands-on experience with LLMs, embeddings, RAG pipelines, or AI agents
- Can design and ship systems quickly — bias toward action over deliberation
- Demonstrates systems thinking and product judgment
- Has built or deployed production AI applications with real, measurable results
- Communicates clearly, writes clean code, owns problems end-to-end

Nice to have:
- Experience with LangChain, HuggingFace, vector databases, or embedding models
- Computer vision or NLP project experience
- Full-stack awareness (APIs, Docker, cloud deployment)
- Startup or fast-paced environment experience

We value: Applied AI fluency, shipping velocity, clarity, ownership.
We do NOT value: pedigree, buzzword-dense resumes with no shipped output.
"""

# ── Scoring Rubric ────────────────────────────────────────────────────────────
#
# Weights chosen to reward the exact profile AI Know A Guy cares about:
#  - Technical depth in AI/ML is the primary signal (55% combined)
#  - Shipping fast is the next biggest signal (20%)
#  - Systems thinking separates senior from mid-level (15%)
#  - Communication quality is a tie-breaker (10%)
#
RUBRIC = {
    "technical_skills": {
        "weight": 0.30,
        "description": (
            "Python proficiency, AI/ML frameworks (PyTorch, TF, HuggingFace), "
            "LLMs, embeddings, APIs. 9-10 = strong AI stack, production experience. "
            "5-6 = solid fundamentals with some AI exposure. <4 = primarily non-AI roles."
        ),
    },
    "ai_ml_depth": {
        "weight": 0.25,
        "description": (
            "Hands-on AI project depth, production deployments, real metrics, "
            "model evaluation experience. 9-10 = shipped AI features with hard numbers. "
            "7-8 = strong projects, measurable outcomes. <5 = theoretical or academic only."
        ),
    },
    "shipping_velocity": {
        "weight": 0.20,
        "description": (
            "Evidence of building and deploying things fast. Startup mindset. "
            "Quantified results, multiple shipped projects, side projects or OSS. "
            "9-10 = serial builder with demonstrated throughput. <5 = slow or unclear output."
        ),
    },
    "systems_thinking": {
        "weight": 0.15,
        "description": (
            "Architecture decisions, tradeoffs, scalability awareness, end-to-end ownership. "
            "9-10 = designed systems from scratch, articulates tradeoffs. "
            "<5 = feature-level thinking, no architectural awareness visible."
        ),
    },
    "clarity_communication": {
        "weight": 0.10,
        "description": (
            "Resume clarity, quantified achievements, signal-to-noise ratio. "
            "9-10 = every bullet has an impact metric, crisp writing. "
            "<5 = vague bullets, filler words, no numbers, poor organization."
        ),
    },
}

# Data Models 

@dataclass
class ResumeScore:
    filename: str
    name: str = "Unknown"
    raw_text_length: int = 0
    rank: int = 0

    # Dimension scores (0-10, LLM-assigned)
    technical_skills: float = 0.0
    ai_ml_depth: float = 0.0
    shipping_velocity: float = 0.0
    systems_thinking: float = 0.0
    clarity_communication: float = 0.0

    # Composite signals
    llm_weighted_score: float = 0.0     # 0-10, weighted by rubric
    embedding_similarity: float = 0.0   # 0-1, cosine sim to JD
    final_score: float = 0.0            # 0-1, fused signal

    # Qualitative output
    strengths: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    one_line_summary: str = ""

    # Error flags
    parse_error: bool = False
    llm_error: bool = False


@dataclass
class PipelineConfig:
    resumes_dir: str = "./resumes"
    output_dir: str = "./output"
    top_pct: float = 0.05           # fraction of pool to surface (default top 5%)
    min_top_n: int = 10             # always surface at least this many
    llm_provider: str = "anthropic"
    # claude-3-5-haiku is fast and cheap: ~$0.02 for 200 resumes at 800 tok/resume
    llm_model: str = "claude-3-5-haiku-20241022"
    max_text_chars: int = 6000      # truncate resumes to control token cost
    batch_delay: float = 0.3        # seconds between API calls (rate limit buffer)
    llm_score_weight: float = 0.70
    embed_score_weight: float = 0.30


# ── PDF / Text Parser ─────────────────────────────────────────────────────────

class ResumeParser:
    """
    Extracts plain text from PDF or TXT resume files.
    Tries pdfplumber first (better multi-column layout handling),
    then PyMuPDF as fallback, then skips with an error flag.
    """

    def parse(self, filepath: Path) -> tuple[str, bool]:
        """Returns (text, had_error)."""
        ext = filepath.suffix.lower()
        if ext == ".txt":
            try:
                return filepath.read_text(encoding="utf-8", errors="ignore"), False
            except Exception:
                return "", True
        elif ext == ".pdf":
            return self._parse_pdf(filepath)
        return "", True

    def _parse_pdf(self, filepath: Path) -> tuple[str, bool]:
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(filepath) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages).strip()
                if text:
                    return text, False
            except Exception:
                pass

        if HAS_FITZ:
            try:
                doc = fitz.open(str(filepath))
                text = "\n".join(page.get_text() for page in doc).strip()
                doc.close()
                if text:
                    return text, False
            except Exception:
                pass

        return "", True


# ── Embedding Engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Computes cosine similarity between a resume and the job description.

    Primary: OpenAI text-embedding-3-small (requires OPENAI_API_KEY).
             Best semantic quality; negligible cost (~$0.001 per 200 resumes).

    Fallback: TF-IDF cosine similarity, fully self-contained, no API needed.
              Good enough to distinguish wildly off-target resumes from matches.
              For production, always use real embeddings.
    """

    # Common English stop words to ignore in TF-IDF
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "i", "we", "you", "he",
        "she", "it", "they", "this", "that", "these", "those", "my", "our",
        "as", "if", "not", "no", "so", "than", "then", "also",
    }

    def __init__(self):
        self._openai_client = None
        self._jd_vec = None
        self._use_openai = False

        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self._openai_client = openai.OpenAI()
            self._use_openai = True
        else:
            print("  [Embedding] No OPENAI_API_KEY — using TF-IDF cosine similarity.")

    def prime(self, jd_text: str):
        """Pre-compute the job description vector once."""
        self._jd_vec = self._vectorize(jd_text)

    def similarity(self, resume_text: str) -> float:
        if self._jd_vec is None:
            return 0.0
        resume_vec = self._vectorize(resume_text)
        return round(self._cosine(self._jd_vec, resume_vec), 4)

    # ── internal ──────────────────────────────────────────────────────────────

    def _vectorize(self, text: str):
        if self._use_openai:
            return self._openai_embed(text)
        return self._tfidf(text)

    def _openai_embed(self, text: str) -> list:
        resp = self._openai_client.embeddings.create(
            input=text[:8000],
            model="text-embedding-3-small",
        )
        return resp.data[0].embedding

    def _tfidf(self, text: str) -> dict:
        """
        Term-frequency dict (with IDF approximated by stop-word removal).
        Returns a sparse vector as {token: tf_score}.
        """
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        tokens = [t for t in tokens if t not in self.STOP_WORDS]
        total = len(tokens) or 1
        freq: dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        # Normalize to term frequency
        return {t: c / total for t, c in freq.items()}

    def _cosine(self, a, b) -> float:
        if self._use_openai and HAS_NUMPY:
            va = np.array(a)
            vb = np.array(b)
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            if denom == 0:
                return 0.0
            raw = float(np.dot(va, vb) / denom)
            # Normalize from [-1, 1] to [0, 1]
            return (raw + 1) / 2

        # TF-IDF dict cosine
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ── LLM Scorer ────────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
You are a senior technical recruiter evaluating candidates for a Senior Software \
Engineer role at an AI startup that ships fast.

JOB DESCRIPTION:
{jd}

CANDIDATE RESUME:
{resume}

Score this candidate on each dimension from 0 to 10.
Be rigorous — reserve 9-10 for genuinely exceptional candidates only.
Base your scores strictly on evidence in the resume text above.

Dimensions:
1. technical_skills  — Python, AI/ML stack, LLMs, production engineering
2. ai_ml_depth       — hands-on AI projects, deployment, real measurable results
3. shipping_velocity — evidence of shipping things fast, startup mindset, output quantity
4. systems_thinking  — architecture, tradeoffs, end-to-end ownership
5. clarity_communication — resume clarity, quantified achievements, signal quality

Respond with ONLY valid JSON, no prose, no markdown fences:
{{
  "name": "<candidate full name, or 'Unknown' if not found>",
  "technical_skills": <integer 0-10>,
  "ai_ml_depth": <integer 0-10>,
  "shipping_velocity": <integer 0-10>,
  "systems_thinking": <integer 0-10>,
  "clarity_communication": <integer 0-10>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
  "gaps": ["<specific gap 1>", "<specific gap 2>"],
  "one_line_summary": "<one punchy, specific sentence about this candidate>"
}}"""


class LLMScorer:
    """
    Scores a resume text against the rubric using an LLM.

    Provider priority:
      1. Anthropic Claude Haiku  (fast, cheap, great instruction-following)
      2. OpenAI GPT-4o-mini      (fallback if no Anthropic key)
      3. Zero scores             (if no keys — still runs, just no LLM signal)
    """

    # Groq free-tier models (get key at console.groq.com — no credit card needed)
    GROQ_MODELS = {
        "groq-llama3-70b": "llama-3.3-70b-versatile",   # best quality, free
        "groq-llama3-8b":  "llama3-8b-8192",             # fastest, free
    }

    def __init__(self, config: PipelineConfig, api_key: str = "", provider: str = ""):
        self.config = config
        self._client_anthropic = None
        self._client_openai = None
        self._client_groq = None   # Groq reuses the OpenAI SDK (same interface)
        self._groq_model = "llama-3.3-70b-versatile"

        # Explicit overrides (from Streamlit UI) take priority over env vars
        _provider = provider or config.llm_provider
        _key = api_key or ""

        if _provider == "groq":
            key = _key or os.getenv("GROQ_API_KEY", "")
            if key and HAS_OPENAI:
                # Groq is OpenAI-compatible — just change base_url
                self._client_groq = openai.OpenAI(
                    api_key=key,
                    base_url="https://api.groq.com/openai/v1",
                )
        elif _provider == "anthropic":
            key = _key or os.getenv("ANTHROPIC_API_KEY", "")
            if key and HAS_ANTHROPIC:
                self._client_anthropic = anthropic.Anthropic(api_key=key)
        elif _provider == "openai":
            key = _key or os.getenv("OPENAI_API_KEY", "")
            if key and HAS_OPENAI:
                self._client_openai = openai.OpenAI(api_key=key)

        # Auto-detect fallback from env if nothing matched above
        if not any([self._client_anthropic, self._client_openai, self._client_groq]):
            if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
                self._client_anthropic = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            elif HAS_OPENAI and os.getenv("GROQ_API_KEY"):
                self._client_groq = openai.OpenAI(
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                )
            elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                self._client_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                print("  [LLM] No API key found. Embedding-only mode.")

    def score(self, resume_text: str) -> dict:
        prompt = _SCORE_PROMPT.format(
            jd=JOB_DESCRIPTION,
            resume=resume_text[: self.config.max_text_chars],
        )
        raw = self._call(prompt)
        return self._parse(raw)

    # ── internal ──────────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        if self._client_anthropic:
            try:
                msg = self._client_anthropic.messages.create(
                    model=self.config.llm_model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception as e:
                return json.dumps({"error": str(e)})

        # Groq (free tier) — same as OpenAI SDK, just different endpoint + model
        if self._client_groq:
            try:
                resp = self._client_groq.chat.completions.create(
                    model=self._groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return json.dumps({"error": str(e)})

        if self._client_openai:
            try:
                resp = self._client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return json.dumps({"error": str(e)})

        # No LLM — return null scores
        return json.dumps({
            "name": "Unknown",
            "technical_skills": 0, "ai_ml_depth": 0, "shipping_velocity": 0,
            "systems_thinking": 0, "clarity_communication": 0,
            "strengths": [],
            "gaps": ["No LLM API key configured — embedding-only scoring active"],
            "one_line_summary": "LLM scoring unavailable",
        })

    def _parse(self, raw: str) -> dict:
        # Strip markdown fences if LLM added them despite instructions
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

