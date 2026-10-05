# Submission notes

## Included

- Interactive Gradio application and root launcher.
- Separate speech-to-text, transcript-refinement, and documentation stages.
- External prompts, dependency manifest, environment example, and setup instructions.
- Technical model/architecture descriptions and evaluator demo walkthrough.
- Unit tests for transcript chunking and retention, protected transcript values, UI output binding, evidence, ownership/deadlines, audio validation, and pipeline stage order.
- Existing shareable recording at `05_DEMO/recording/bootcamp_test_meeting.wav`.
- Actual end-to-end outputs in `05_DEMO/outputs/`, generated from the included recording using local faster-whisper and both configured Groq stages.

## Demo output status

The sample recording was processed through the application. The static demo outputs are copies of the successful run under `01_APPLICATION/outputs/`; the JSON, Markdown, decisions, and action-item files correspond to that same validated run. The raw and refined transcript files happen to match because the refinement stage made no safe changes to this recording. No output was fabricated.

## Credentials and generated files

Never include `.env` or real API keys. Runtime outputs are written under `01_APPLICATION/outputs/` and ignored by Git. The curated demo artifacts in `05_DEMO/outputs/` are intentionally included for review; share them only if the recording and transcript are permitted for distribution.
