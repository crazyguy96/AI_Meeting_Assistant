# AI Meeting Assistant — Inter IIT Bootcamp Phase 2

## 1. Project overview

This application accepts a meeting audio file and produces a raw transcript, a terminology-aware refined transcript, and an evidence-checked meeting record. The Gradio interface presents the complete raw and refined transcripts in separate views, plus minutes, decisions, action items, and the complete final record as JSON and Markdown. It also offers a ZIP download containing transcripts, minutes, decisions, action items, records, a refinement audit, and the uploaded audio.

The system retains the raw speech-to-text transcript unchanged as a separate output. Transcript correction is a distinct LLM stage from meeting documentation.

## 2. Architecture

```text
Audio upload → speech-to-text → retained raw transcript
  → token-budgeted terminology extraction + ordered transcript refinement
  → protected-information edit checks → refined transcript
  → meeting-documentation LLM → proposed meeting record
  → evidence, owner, and deadline validation
  → JSON + Markdown + ZIP → Gradio inspection/download
```

## 3. Models used

| Stage | Model | Purpose |
|---|---|---|
| Speech-to-text | Local faster-whisper `small` | Transcribes English audio locally using CPU and int8 compute. |
| LLM stage 1 | Groq `openai/gpt-oss-120b` | Extracts terminology and proposes conservative transcript corrections. |
| LLM stage 2 | Groq `openai/gpt-oss-20b` | Converts the refined transcript into minutes, decisions, non-decisions, and action items. |

The local faster-whisper model returns timestamped transcript segments. The same first-stage Groq model does terminology extraction and refinement. Long inputs are split at transcript/text boundaries using the `o200k_base` tokenizer; all chunks are processed in order and reassembled without dropping source text. Python checks proposed changes for protected numbers, money, dates, negation, and modality. A separate validator grounds decisions and action items in transcript evidence and removes unsupported owner/deadline claims.

## 4. Folder structure

```text
01_APPLICATION/              Application, pipeline, shared code, and launcher
02_PROMPTS/                  External model prompts
03_DEPENDENCIES/             Requirements, environment example, setup
04_README/                    Evaluator-oriented project documentation
05_DEMO/                      Shareable recording and real run outputs, when generated
06_TECHNICAL_DESCRIPTION/     Model roles and architecture
07_TESTS/                     Safety, evidence, and pipeline tests
08_SUBMISSION/                End-to-end evaluator instructions
run.py                        Root-level launcher
.gitignore                    Secrets and generated-file exclusions
submission_notes.md           Submission status and remaining steps
```

## 5. Installation

Requires Python 3.10 or newer; Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r 03_DEPENDENCIES/requirements.txt
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

The faster-whisper requirement is pinned to the upstream PyAV 19 fix. If updating an existing environment that already has PyPI `faster-whisper==1.2.1`, see `03_DEPENDENCIES/environment_setup.md` for the one-time force-reinstall command.

## 6. Environment variables

Create `.env` from `03_DEPENDENCIES/.env.example` in the repository root if it is absent, then configure:

- `GROQ_API_KEY` — Groq refinement and documentation access

No OpenAI API key is required. Optional Groq model overrides are documented in the example file. Never commit `.env`.

## 7. Run the application

```bash
python run.py
```

Use the Gradio URL printed in the terminal. Upload an audio recording, optionally provide comma-separated technical terms/names, and select **Start processing**. After completion, inspect the separate Raw Transcript, Refined Transcript, Minutes, Decisions, Action Items, and Meeting Record views, or download the generated files.

To run only the included recording through local faster-whisper (no Groq calls), use:

```bash
python -c "import sys; sys.path.insert(0, '01_APPLICATION'); from app.pipeline.transcription import transcribe_audio; result = transcribe_audio('05_DEMO/recording/bootcamp_test_meeting.wav'); print(result['text']); print(result['segments'])"
```

## 8. Run tests

```bash
python -m pytest 07_TESTS
```

Tests use mocks for external API calls and do not require API keys or send audio to model providers.

## 9. Demo workflow

Use `05_DEMO/recording/bootcamp_test_meeting.wav` in the running UI. Inspect each transcript and the generated record, then use the ZIP download. See [DEMO.md](../08_SUBMISSION/DEMO.md) for the detailed evaluator walkthrough.

The sample recording is included. `05_DEMO/outputs/` is populated only with artifacts copied from a successfully completed application run; generated run artifacts are not fabricated.

## 10. Expected outputs

Each successful run produces:

- `raw_transcript.txt`
- `refined_transcript.txt`
- `minutes.md`
- `decisions.json`
- `action_items.json`
- `meeting_record.json`
- `meeting_record.md`
- `refinement_audit.json`
- a copy of the uploaded audio
- a ZIP archive containing those files

## 11. Known limitations

- Speech recognition runs locally after the faster-whisper model has been downloaded; only the two Groq LLM stages require network access and a valid Groq API key.
- Transcription provides segment timestamps but does not identify speakers.
- Speaker identity is not inferred. Owners and deadlines are retained only when explicitly evidenced.
- Evidence matching uses text similarity and may require human review; the application deliberately drops unsupported decisions and marks uncertain action items for review.
- The supplied sample audio has not been processed into demo outputs in this checkout.
