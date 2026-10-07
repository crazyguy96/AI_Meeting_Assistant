# 🎙️ AI Meeting Assistant

> From raw meeting audio to a grounded, structured, and actionable meeting record.

An end-to-end multimodel meeting pipeline that converts recorded English meeting audio into:

**Raw Transcript → Refined Transcript → Evidence-validated Meeting Record**

The core principle is:

> **Preserve First. Improve Second. Extract Third.**

---

## ⚡ Quick Start

### 1. Clone and enter the repository

```bash
git clone https://github.com/crazyguy96/AI_Meeting_Assistant.git
cd AI_Meeting_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r 03_DEPENDENCIES/requirements.txt
```

### 4. Configure Groq

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit the real `.env` or an API key.

### 5. Run

```bash
python run.py
```

The Gradio interface will start locally.

### 6. Run tests

```bash
pytest 06_TESTS
```

**Current verification: 98 tests passing.**

---

## 🎯 What Problem Does It Solve?

Important meeting information is usually buried inside long recordings:

- technical discussions
- decisions
- commitments
- action items
- owners
- deadlines
- unresolved issues

A simple LLM summary is risky because it can turn a proposal into a decision or a suggestion into an assignment.

This system is designed around a stricter question:

> **How can we improve and structure a meeting conversation without inventing information that was never said?**

---

## 🧠 Core Architecture

```text
Meeting Audio
     ↓
Input Validation
     ↓
faster-whisper
     ↓
RAW TRANSCRIPT
     ↓
GPT-OSS-120B
     ↓
REFINED TRANSCRIPT
     ↓
GPT-OSS-20B
     ↓
Evidence Validation
     ↓
FINAL MEETING RECORD
```

### Stage 1 — Speech Recognition

**faster-whisper** performs local English speech-to-text and produces timestamped segments.

```text
Meeting Audio
     ↓
faster-whisper
     ↓
Timestamped Segments
     ↓
Raw Transcript
```

The raw transcript is preserved as its own artifact rather than silently replacing it with an LLM-generated version.

### Stage 2 — Domain-Aware Refinement

**GPT-OSS-120B via Groq** improves transcript quality while preserving meaning.

It focuses on:

- technical terminology
- names and acronyms
- numbers and monetary values
- negation
- commitments
- grammar and readability
- terminology consistency

It is **not** responsible for producing the final meeting summary.

### Stage 3 — Structured Documentation

**GPT-OSS-20B via Groq** receives the refined transcript and extracts:

- summary
- confirmed decisions
- action items
- owners when explicitly stated
- deadlines when explicitly stated
- supporting evidence

### Evidence Validation

The final extraction is checked against transcript evidence.

The system explicitly distinguishes:

```text
Proposal        ≠ Decision
Suggestion      ≠ Assignment
Mentioned date  ≠ Deadline
Possibility     ≠ Commitment
```

For example:

```text
"We could move this to next week."
        ↓
Proposal — not automatically a decision
```

and:

```text
"Maybe Jatin can handle it."
        ↓
Suggestion — not automatically an assignment
```

---

## 🔒 Safety & Reliability

The project follows a simple rule:

> **A meeting assistant should fail explicitly rather than silently invent information.**

### Refinement safety

Important information is protected during refinement:

- names
- numbers
- monetary values
- negation
- commitments
- ordering
- technical terminology

For example:

```text
"We will not deploy this version."
```

must not become:

```text
"We will deploy this version."
```

### Evidence-first extraction

Only information supported by the conversation should become a confirmed decision or action.

### Optional stages fail gracefully

Glossary extraction is optional. If it fails:

```text
Glossary Failure
      ↓
Warning / Fallback
      ↓
Continue Refinement
```

Required stages such as refinement and documentation fail explicitly rather than silently generating an incomplete record.

---

## 🧩 Long-Meeting Processing

Long transcripts can exceed the practical context budget of a single request.

The refinement stage therefore uses token-aware chunking:

```text
Long Transcript
      ↓
Token-aware Chunking
      ↓
Chunk 1   Chunk 2   ...   Chunk N
      ↓
Sequential Refinement
      ↓
Ordered Reconstruction
      ↓
Refined Transcript
```

This avoids relying on a hardcoded meeting duration.

---

## 🔁 Structured Output Reliability

Structured LLM calls use several protections:

- strict JSON schemas
- prompt/schema alignment
- explicit empty-array handling
- JSON recovery
- retries with backoff
- fallback handling
- reasoning-token controls for supported reasoning models
- API-key sanitization

The goal is reliable structured generation rather than simply generating fluent text.

---

## 📡 Pipeline Observability

A successful run exposes explicit stages:

```text
received
validated
transcribing
raw_ready
refining
refined
documenting
validating
evidence
exporting
complete
```

For long recordings, stage-level status makes failures diagnosable instead of reducing everything to a generic "Processing failed."

---

## 🎨 User Interface

The Gradio interface is organized around the three semantic representations:

### Raw Transcript
The direct speech-recognition output.

### Refined Transcript
The domain-aware transcript after controlled refinement.

### Final Output
The structured meeting record containing:

- summary
- decisions
- action items
- owners
- deadlines
- evidence

The intended workflow is:

```text
Upload → Process → Inspect → Export
```

The UI also provides stage-level status, readable transcript views, error presentation, responsive layout, and downloadable outputs.

---

## 🧪 Testing

The project includes a dedicated test suite under `06_TESTS/`.

Run:

```bash
pytest 06_TESTS
```

Current verification:

```text
43 passed
```

Coverage includes:

| Test | Focus |
|---|---|
| `test_pipeline.py` | validation, pipeline execution, outputs |
| `test_refinement.py` | chunking, ordering, safety, retries, fallback |
| `test_safety.py` | negation, monetary values, semantic preservation |
| `test_evidence.py` | evidence matching and decision classification |
| `test_ui.py` | status streaming, errors, output layout |

---

## 📊 End-to-End Verification

A real approximately 15-minute meeting was processed successfully through the complete pipeline.

Verified output:

```text
Raw transcript      → 14,073 characters
Refined transcript  → 14,104 characters
Confirmed decisions → 2
Action items        → 4
```

The run reached:

```text
received → validated → transcribing → raw_ready
→ refining → refined → documenting → validating
→ evidence → exporting → complete
```

---

## 🎬 Demo

All demo material is under:

```text
04_DEMO/
├── recording/
└── outputs/
```

The repository contains recordings and complete generated artifacts for the demo meetings.

Each output bundle can contain:

```text
raw_transcript.txt
refined_transcript.txt
refinement_audit.json
decisions.json
action_items.json
minutes.md
meeting_record.json
meeting_record.md
```

See `07_SUBMISSION/DEMO.md` for the evaluator-facing demo workflow.

---

## 📦 Output Contract

| Output | Purpose |
|---|---|
| `raw_transcript.txt` | Original speech-recognition output |
| `refined_transcript.txt` | Domain-aware refined transcript |
| `refinement_audit.json` | Refinement audit information |
| `minutes.md` | Human-readable meeting minutes |
| `decisions.json` | Machine-readable confirmed decisions |
| `action_items.json` | Machine-readable action items |
| `meeting_record.json` | Complete structured record |
| `meeting_record.md` | Human-readable final record |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Speech-to-Text | faster-whisper |
| Refinement | `openai/gpt-oss-120b` |
| Documentation | `openai/gpt-oss-120b` |
| LLM Inference | Groq |
| UI | Gradio |
| Structured Output | JSON / JSON Schema |
| Testing | Pytest |

### Why faster-whisper?

Local STT keeps transcription independent from the hosted LLM stages and provides timestamped segments without requiring a separate hosted transcription quota.

### Why Groq?

Groq provides fast inference for the language-model stages, which is useful for an interactive multi-stage pipeline.

### Why two language models?

The models have deliberately separated responsibilities:

```text
GPT-OSS-120B
      ↓
Transcript Refinement
      ↓
GPT-OSS-20B
      ↓
Documentation & Extraction
```

This makes the pipeline easier to inspect and helps identify where an error was introduced.

---

## 🧱 Code Structure

```text
AI_Meeting_Assistant/
│
├── 01_APPLICATION/
│   └── app/
│       ├── core/
│       │   ├── clients.py
│       │   ├── config.py
│       │   └── utils.py
│       ├── pipeline/
│       │   ├── transcription.py
│       │   ├── refinement.py
│       │   ├── documentation.py
│       │   └── evidence.py
│       └── ui/
│           └── gradio_app.py
│
├── 02_PROMPTS/
│   ├── glossary_prompt.txt
│   ├── refinement_prompt.txt
│   └── documentation_prompt.txt
│
├── 03_DEPENDENCIES/
│   ├── requirements.txt
│   └── environment_setup.md
│
├── 04_DEMO/
│   ├── recording/
│   └── outputs/
│
├── 05_TECHNICAL_DESCRIPTION/
├── 06_TESTS/
├── 07_SUBMISSION/
├── README.md
└── run.py
```

### Core modules

- `transcription.py` — audio processing and faster-whisper inference
- `refinement.py` — glossary, chunking, refinement, retries and reconstruction
- `documentation.py` — structured meeting-record generation
- `evidence.py` — evidence and extraction validation
- `clients.py` — LLM client abstraction
- `config.py` — model and environment configuration
- `gradio_app.py` — application interface

---

## 📝 Prompt Architecture

Prompts are kept separately under `02_PROMPTS/`:

```text
glossary_prompt.txt
refinement_prompt.txt
documentation_prompt.txt
```

This keeps prompt engineering separate from application logic and makes model behavior easier to inspect and version.

---

## 🔐 Environment & Security

The repository uses environment variables for credentials.

Use:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit:

```text
.env
```

or any file containing a live API credential.

The application also sanitizes API-key-like values before logging exceptions.

---

## 📚 Documentation

Additional technical material is available in:

```text
05_TECHNICAL_DESCRIPTION/
07_SUBMISSION/
```

The repository includes the architecture, technical description, test suite, demo artifacts, and evaluator-facing submission material.

---

## 🚀 Design Summary

The system is intentionally built around traceability:

```text
Audio
  ↓
Raw Transcript
  ↓
Refined Transcript
  ↓
Structured Record
  ↓
Evidence Validation
  ↓
Actionable Output
```

The central design principle remains:

> **Preserve First. Improve Second. Extract Third.**

The result is a multimodel meeting assistant designed to improve transcript quality and meeting documentation **without losing the distinction between what was said, what was refined, and what was actually decided.**
