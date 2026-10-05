# Technical description

## Models and roles

| Stage | Model | Role |
|---|---|---|
| Speech-to-text | Local faster-whisper `small` | Transcribes English audio locally on CPU with int8 compute and returns timestamped segments. |
| First language-model stage | Groq `openai/gpt-oss-120b` | Extracts a meeting terminology glossary, then proposes high-confidence recognition corrections against exact transcript segments. |
| Second language-model stage | Groq `openai/gpt-oss-20b` | Produces a structured meeting record from the refined transcript. |

The Groq model names are configurable through environment variables; their defaults preserve the existing project roles. faster-whisper uses the configured local model/device/compute type: `small`, `cpu`, and `int8`.

## Data flow and stage contracts

1. **Audio input.** Gradio supplies a local filepath. The application rejects unsupported extensions, missing/non-file paths, zero-length files, and files that cannot be read before transcription.
2. **Speech-to-text.** A cached faster-whisper `WhisperModel` runs locally with `language="en"`. Its real segment start/end times are retained alongside each segment ID and text. Segment text is joined for the downstream transcript field. The raw text and timestamped segments are retained independently and are never overwritten by refinement.
3. **Terminology and refinement (LLM 1).** The first Groq model extracts terms and combines them with any user-provided glossary. A `tiktoken` `o200k_base` encoding measures prompt size. Glossary input and refinement segments are split into ordered requests within a 5,200-token input budget (including prompt text and a chat-overhead reserve). Oversized individual segments are split into exact text fragments for requests, then edits are applied by their original segment ID to the untouched full segment list. The application reassembles only by retaining the original segment sequence; raw transcript data is never mutated or truncated.
4. **Protected-information checks.** Python accepts only edits whose exact source occurs in the requested segment, whose confidence is at least 0.75, and whose protected tokens remain unchanged. Tokens cover numeric values, monetary amounts, percentages, dates/day names, negation, and modality. Rejected proposals remain available in the refinement audit.
5. **Documentation (LLM 2).** A separate Groq model receives only the refined transcript and returns the concise summary, minutes, decisions, non-decisions, action items, discussion points, and open questions. Prompts require evidence quotes and prohibit inferred owners/deadlines or converting proposals into decisions.
6. **Evidence validation.** Python checks decision evidence against refined transcript segments and excludes unsupported decisions. Action items are retained for visibility but marked `Needs Review` if their evidence is unsupported. Evidence scores, timestamps, and segment IDs are recorded.
7. **Owner/deadline validation.** A non-`Unspecified` owner and deadline must appear in the action's evidence quote; otherwise that field is reset to `Unspecified` and status becomes `Needs Review`. Missing values are represented explicitly as `Unspecified`.
8. **Outputs.** The application writes raw/refined transcripts, minutes, decisions, action items, a validated meeting-record JSON file, Markdown rendering, and a refinement audit to a unique runtime directory. The Gradio UI binds each transcript to its own complete viewer, fills the dedicated Decisions and Action Items tabs, and displays the full final record separately in JSON and Markdown. It adds the uploaded audio and packages the files in a downloadable ZIP.

## Error handling

Missing Groq API keys, unsupported or unreadable audio, empty audio, malformed model JSON, local model loading/transcription errors, provider errors, and missing/empty prompt files raise explicit errors. The UI shows the failed stage and technical detail; transcripts successfully produced before a later-stage failure remain available as partial results, while the final record is explicitly marked incomplete. Completion is signaled only after validation and export finish. Local speech-to-text requires no OpenAI key.

## Privacy and reproducibility

Audio is transcribed locally by faster-whisper. Transcript text is sent to Groq for refinement and meeting documentation. Users should avoid sending sensitive transcripts to a provider unless permitted by their organization. Tests mock model calls and do not send data externally. Generated files are stored under `01_APPLICATION/outputs/` and excluded from source control.
