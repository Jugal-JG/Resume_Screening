# How the Resume Screening System Works

### The Problem

AI Know A Guy received 200+ job applications. They have no HR team and need to ship a real AI product in 30 days. Manually reading 200 resumes would take ~2 full days of human time and introduce subjective bias. The challenge: **find the top 5% automatically, using AI**.

---

### The Solution in Plain English

Think of this system as a **smart resume reader that asks two questions about every applicant**:

1. **Does this resume talk about the right things?**
   A language model (like ChatGPT, but cheaper and faster) reads each resume and gives it a score on 5 specific qualities that matter for this role.

2. **Does this resume sound similar to the job description?**
   The system converts both the job description and the resume into a "fingerprint" (a list of numbers that captures what topics they cover), then measures how close the two fingerprints are.

These two scores are combined 70% from question 1, 30% from question 2 into a single number from 0 to 100. Resumes are ranked by that number. The top 5% (roughly 10 candidates out of 200) get surfaced for human review.

---

### The 5 Things Being Scored

Every resume is evaluated on exactly five dimensions. The weights reflect what AI Know A Guy actually cares about:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| **Technical Skills** | 30% | Python, AI/ML tools (PyTorch, LangChain, LLMs), production engineering |
| **AI/ML Depth** | 25% | Did they actually build AI things? Do they have real numbers to show for it? |
| **Shipping Velocity** | 20% | Do they have a bias toward action? Have they shipped multiple things? |
| **Systems Thinking** | 15% | Can they design systems, not just write features? Do they understand tradeoffs? |
| **Clarity** | 10% | Is the resume clear? Are achievements quantified? Is there signal, not noise? |

Each dimension is scored 0–10 by the language model. These are then combined using the weights above to get the LLM score. The semantic similarity score (question 2) is added as a 30% bonus signal.

---

### What the System Does NOT Do

- It does not make hiring decisions. It surfaces candidates for human review.
- It does not read between the lines. Scores are based strictly on what's written.
- It does not filter by school, location, or any demographic signal.
- It does not replace a phone screen. It just makes the shortlist faster.

---
### The Two Components

**1. Embedding Engine** (`EmbeddingEngine` class)

"Embeddings" are just a way to convert text into a list of numbers that capture meaning. Two texts that talk about similar things will have numbers that are close together. This system uses a free, built-in method called TF-IDF (counts how often important words appear relative to all words) as the default. If you provide an OpenAI API key, it upgrades to a proper neural embedding model (`text-embedding-3-small`) for better accuracy.

**2. LLM Scorer** (`LLMScorer` class)

Sends a carefully written prompt to a language model that says: *"Here is the job description. Here is the resume. Score it on these 5 dimensions and explain why."* The model responds with a JSON object containing the scores, strengths, gaps, and a one-line summary. Temperature is set to 0 (no randomness) so scores are consistent.

---

### How to Get a Free API Key (No Credit Card)

**Recommended: Groq (completely free)**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google — no credit card needed
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with `gsk_...`)
5. Paste it into the app sidebar, select "groq" as the provider
6. This gives you access to Llama 3 (Meta's open-source model), which is excellent for scoring

**Alternative: Google AI Studio (free)**
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with Google
3. Click "Get API key" → "Create API key"
4. *(Note: the current app supports Groq and Anthropic/OpenAI Gemini support is easy to add)*

**Alternative: Anthropic (free credits on signup)**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. New accounts get $5 in free credits enough for ~10,000 resume scorings
3. Get key from "API Keys" section

---

### How to Run It

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app
streamlit run streamlit_app.py

```

---

### Tradeoffs and Known Limitations

| Risk | What could go wrong | How it's handled |
|------|---------------------|-----------------|
| **LLM hallucination** | Model invents skills not in the resume | Temperature=0, prompt says "base scores strictly on resume text only" |
| **Bad PDF parsing** | Multi-column resumes lose structure | Falls back to PyMuPDF; candidates can paste text instead |
| **Keyword stuffing** | Applicant copies job description into resume | LLM depth scoring asks for *evidence and metrics*, not just keyword presence |
| **No API key** | Can't score without a key | Falls back to embedding-only mode still ranks by semantic similarity |
| **Cost at scale** | 200 resumes × 800 tokens = 160K tokens | Groq is free; Anthropic Haiku costs ~$0.02 total |

---
