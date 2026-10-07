import html
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from app.core.utils import render_markdown, sanitize_exception
from app.main import process_meeting
from app.pipeline.transcription import SUPPORTED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

# Ordered pipeline stages with user-facing labels and descriptions for the UI stepper.
STAGES = (
    ("received", "Audio received", "The uploaded recording is ready to validate."),
    ("validated", "Audio validated", "File type, location, and readability checked."),
    ("transcribing", "Transcribing meeting", "Running local Whisper transcription."),
    ("raw_ready", "Raw transcript ready", "The complete timestamped Whisper transcript is retained."),
    ("refining", "Refining terminology", "Checking domain terms and transcript edits."),
    ("refined", "Refined transcript ready", "The complete refined transcript is retained."),
    ("documenting", "Generating meeting record", "Creating minutes, decisions, and tasks."),
    ("validating", "Validating evidence", "Checking decisions and action evidence."),
    ("exporting", "Preparing downloads", "Writing the meeting result files."),
    ("complete", "Processing complete", "Your meeting results are ready."),
)
STAGE_DETAILS = {stage_id: detail for stage_id, _, detail in STAGES}

# Labels and filenames for the individual deliverables generated in each run.
OUTPUT_FILES = (
    ("Meeting record · JSON", "meeting_record.json"),
    ("Meeting record · Markdown", "meeting_record.md"),
    ("Raw transcript", "raw_transcript.txt"),
    ("Refined transcript", "refined_transcript.txt"),
    ("Decisions", "decisions.json"),
    ("Action items", "action_items.json"),
    ("Minutes", "minutes.md"),
)

# Custom light-theme styling ensuring high contrast, clean typography, and responsive layouts.
CSS = """
:root, html, body, .dark, [data-theme="dark"], gradio-app, gradio-app.dark, .gradio-container, .gradio-container.dark {
  color-scheme: light !important;
  --page: #FFFFFF !important;
  --surface: #FFFFFF !important;
  --ink: #111827 !important;
  --muted: #6B7280 !important;
  --line: #D1D5DB !important;
  --accent: #2563EB !important;
  --accent-dark: #1D4ED8 !important;
  --success: #16A34A !important;
  --danger: #DC2626 !important;

  /* Gradio Internal Theme Variables */
  --body-text-color: #111827 !important;
  --body-text-color-subdued: #4B5563 !important;
  --block-label-text-color: #111827 !important;
  --block-title-text-color: #111827 !important;
  --block-info-text-color: #6B7280 !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F9FAFB !important;
  --block-background-fill: #FFFFFF !important;
  --panel-background-fill: #FFFFFF !important;
  --input-background-fill: #FFFFFF !important;
  --input-placeholder-color: #6B7280 !important;
  --border-color-primary: #D1D5DB !important;
  --border-color-accent: #2563EB !important;
  --table-text-color: #111827 !important;
  --button-secondary-text-color: #111827 !important;
  --button-secondary-background-fill: #F9FAFB !important;
  --button-secondary-border-color: #D1D5DB !important;
  --color-accent: #2563EB !important;
  --tw-prose-body: #111827 !important;
  --tw-prose-headings: #111827 !important;
  --tw-prose-lead: #4B5563 !important;
  --tw-prose-links: #2563EB !important;
  --tw-prose-bold: #111827 !important;
  --tw-prose-counters: #4B5563 !important;
  --tw-prose-bullets: #4B5563 !important;
  --tw-prose-hr: #D1D5DB !important;
  --tw-prose-quotes: #111827 !important;
  --tw-prose-quote-borders: #D1D5DB !important;
  --tw-prose-captions: #6B7280 !important;
  --tw-prose-code: #111827 !important;
  --tw-prose-pre-code: #111827 !important;
  --tw-prose-pre-bg: #F9FAFB !important;
  --tw-prose-th-borders: #D1D5DB !important;
  --tw-prose-td-borders: #D1D5DB !important;
}

html, body, gradio-app, gradio-app > .main,
.gradio-container, .gradio-container .main {
  color-scheme: light !important;
  background-color: #FFFFFF !important;
  color: #111827 !important;
  overflow-x: hidden !important;
}

body, .gradio-container {
  background: #FFFFFF !important;
  color: #111827 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.gradio-container {
  max-width: 1040px !important;
  width: 100% !important;
  box-sizing: border-box !important;
  margin: 0 auto !important;
  overflow-x: hidden !important;
}

#app-shell {
  width: 100% !important;
  gap: 20px !important;
  padding: 24px 16px 48px !important;
  background-color: #FFFFFF !important;
  box-sizing: border-box !important;
}

#app-header {
  width: 100% !important;
  border-bottom: 1px solid #D1D5DB !important;
  padding: 4px 0 20px !important;
  display: block !important;
}

#app-header > div {
  width: 100% !important;
}

.header-container {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  width: 100% !important;
  box-sizing: border-box !important;
  gap: 28px !important;
}

.header-left {
  display: flex !important;
  flex-direction: column !important;
  gap: 10px !important;
  min-width: 0 !important;
  flex: 1 1 auto !important;
}

.header-brand-row {
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
}

#brand-mark {
  width: 38px !important;
  height: 38px !important;
  border-radius: 8px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: #EFF6FF !important;
  color: #2563EB !important;
  font-weight: 700 !important;
  border: 1px solid #BFDBFE !important;
  font-size: 18px !important;
  flex-shrink: 0 !important;
}

.brand-statement {
  color: #111827 !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  line-height: 1.4 !important;
  margin: 0 !important;
}

.header-workflow {
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  flex-wrap: wrap !important;
}

.workflow-step {
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  background: #F9FAFB !important;
  border: 1px solid #E5E7EB !important;
  border-radius: 6px !important;
  padding: 4px 10px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  color: #374151 !important;
  line-height: 1 !important;
}

.workflow-icon {
  color: #2563EB !important;
  flex-shrink: 0 !important;
}

.workflow-arrow {
  color: #9CA3AF !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  user-select: none !important;
}

.header-right {
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  text-align: right !important;
  align-items: flex-end !important;
  flex-shrink: 0 !important;
}

.brand-title {
  color: #111827 !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  line-height: 1.25 !important;
  letter-spacing: -0.01em !important;
}

.brand-subtitle {
  color: #4B5563 !important;
  font-size: 13px !important;
  margin-top: 3px !important;
  line-height: 1.3 !important;
}

.hero { text-align: center; padding: 32px 12px 18px; }
.hero h1 { color: #111827 !important; font-size: 32px; font-weight: 700; line-height: 1.2; margin: 0 0 10px; }
.hero p { color: #4B5563 !important; font-size: 15px; max-width: 540px; margin: 0 auto; line-height: 1.5; }

.upload-panel {
  background: #F9FAFB !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 12px !important;
  padding: 24px !important;
  max-width: 720px !important;
  margin: 12px auto 0 !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

.upload-panel h3 { color: #111827 !important; font-size: 16px !important; font-weight: 600 !important; margin: 0 0 4px !important; }
.upload-panel p { color: #4B5563 !important; font-size: 13px !important; margin: 0 0 12px !important; }

/* Audio upload container */
.upload-audio > div {
  border: 1px dashed #D1D5DB !important;
  border-radius: 8px !important;
  background: #FFFFFF !important;
}
.upload-audio span, .upload-audio p, .upload-audio label {
  color: #4B5563 !important;
}

label, label span, .block label, .block label span, .block-label {
  color: #111827 !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}

/* Text inputs */
input[type="text"], input[type="text"]:focus, .textbox input {
  background-color: #FFFFFF !important;
  color: #111827 !important;
  border: 1px solid #D1D5DB !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Buttons */
.primary-cta button, button.primary {
  background: #2563EB !important;
  border: 1px solid #2563EB !important;
  border-radius: 8px !important;
  min-height: 44px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.primary-cta button:hover, button.primary:hover {
  background: #1D4ED8 !important;
}
button.secondary, button:not(.primary):not(.primary-cta button):not([role="tab"]):not(.tab-container button):not([data-tab-id]) {
  background: #FFFFFF !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 8px !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  font-weight: 600 !important;
}
button.secondary:hover, button:not(.primary):not(.primary-cta button):not([role="tab"]):not(.tab-container button):not([data-tab-id]):hover {
  background: #F9FAFB !important;
}

/* Progress panel */
.progress-panel {
  background: #FFFFFF !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 10px !important;
  padding: 16px 20px !important;
  margin: 8px 0 !important;
}
.progress-heading { color: #111827 !important; font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.progress-copy { color: #4B5563 !important; font-size: 13px; margin-bottom: 12px; }
.stage-list { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
.stage {
  display: flex; gap: 8px; align-items: center; min-height: 38px;
  padding: 8px 10px; border-radius: 6px; background: #F9FAFB !important;
  color: #4B5563 !important; font-size: 12px; font-weight: 500;
  border: 1px solid #E5E7EB !important;
}
.stage span { color: inherit !important; }
.stage-mark {
  flex: 0 0 18px; width: 18px; height: 18px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid #D1D5DB; color: #6B7280; font-size: 10px; font-weight: 700;
}
.stage.done { color: #16A34A !important; background: #F0FDF4 !important; border-color: #BBF7D0 !important; }
.stage.done .stage-mark { border-color: #86EFAC; color: #16A34A; background: #FFFFFF; }
.stage.active { color: #2563EB !important; border: 1px solid #BFDBFE !important; background: #EFF6FF !important; font-weight: 600; }
.stage.active .stage-mark { border-color: #2563EB; color: #FFFFFF; background: #2563EB; }

/* Error panel */
.error-panel {
  border: 1px solid #FECACA; background: #FEF2F2; border-radius: 8px;
  padding: 16px; margin: 12px 0; color: #DC2626;
}
.error-title { font-weight: 600; font-size: 15px; margin-bottom: 4px; color: #DC2626 !important; }
.error-message { color: #991B1B !important; font-size: 13px; line-height: 1.5; }

/* Results heading */
.results-heading {
  align-items: center; justify-content: space-between;
  border-bottom: 1px solid #D1D5DB; padding: 4px 0 16px; margin-bottom: 16px;
}
.results-heading h2 { margin: 0; font-size: 22px; font-weight: 700; color: #111827 !important; }

/* Tabs Navigation - High Contrast and Clear Distinction */
.tabs { border: 0 !important; margin-top: 8px !important; }
.tab-nav, .tabs > .tab-nav, div[role="tablist"], .tab-container {
  border-bottom: 2px solid #D1D5DB !important;
  background-color: transparent !important;
  gap: 4px !important;
}

/* Inactive tabs: prominent, dark gray (#4B5563), never faint */
.tab-nav button, .tabs button, button[role="tab"], .tab-container button, button[data-tab-id] {
  color: #4B5563 !important;
  -webkit-text-fill-color: #4B5563 !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 18px !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  opacity: 1 !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -2px !important;
  cursor: pointer !important;
}
.tab-nav button:hover, .tabs button:hover, button[role="tab"]:hover, .tab-container button:hover, button[data-tab-id]:hover {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  background-color: #F9FAFB !important;
}

/* Active tab: primary blue (#2563EB), bold, colored underline */
.tab-nav button.selected, .tabs button.selected, button[role="tab"][aria-selected="true"], .tab-container button.selected, button[data-tab-id].selected {
  color: #2563EB !important;
  -webkit-text-fill-color: #2563EB !important;
  border-bottom: 2px solid #2563EB !important;
  font-weight: 700 !important;
  background-color: transparent !important;
}

/* Transcript Textboxes */
.transcript-box textarea,
.transcript-box textarea:focus,
.transcript-box textarea:disabled {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
  font-size: 13px !important;
  line-height: 1.7 !important;
  background-color: #FFFFFF !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 8px !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  padding: 14px !important;
  opacity: 1 !important;
}

/* Final Output Card: Force ALL text to deep readable colors */
.final-output-card {
  background-color: #FFFFFF !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 8px !important;
  padding: 24px !important;
  line-height: 1.7 !important;
  color: #111827 !important;
  min-height: 400px !important;
}

/* Explicit overrides for all markdown elements inside final-output-card and prose */
.final-output-card *,
.final-output-card h1, .final-output-card h2, .final-output-card h3, .final-output-card h4,
.final-output-card p, .final-output-card li, .final-output-card ul, .final-output-card ol,
.final-output-card strong, .final-output-card em, .final-output-card blockquote,
.final-output-card span, .final-output-card a,
.prose, .prose * {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

.final-output-card h1 {
  font-size: 22px !important;
  font-weight: 700 !important;
  border-bottom: 1px solid #D1D5DB !important;
  padding-bottom: 10px !important;
  margin-top: 0 !important;
  margin-bottom: 16px !important;
  color: #111827 !important;
}

.final-output-card h2 {
  font-size: 17px !important;
  font-weight: 700 !important;
  margin-top: 24px !important;
  margin-bottom: 10px !important;
  color: #111827 !important;
}

.final-output-card p {
  font-size: 14px !important;
  line-height: 1.7 !important;
  color: #111827 !important;
  margin-bottom: 14px !important;
}

.final-output-card ul, .final-output-card ol {
  padding-left: 24px !important;
  margin: 10px 0 16px !important;
}

.final-output-card li {
  font-size: 14px !important;
  line-height: 1.65 !important;
  margin-bottom: 6px !important;
  color: #111827 !important;
}

.final-output-card strong {
  font-weight: 700 !important;
  color: #111827 !important;
}

/* Mobile responsiveness (~589px) and overflow prevention */
@media (max-width: 640px) {
  #app-shell {
    padding: 12px 8px 32px !important;
    gap: 14px !important;
  }
  .header-container {
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 16px !important;
  }
  .header-right {
    text-align: left !important;
    align-items: flex-start !important;
  }
  .header-workflow {
    flex-wrap: wrap !important;
    gap: 6px !important;
  }
  .brand-statement {
    font-size: 13px !important;
  }
  .stage-list {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 6px !important;
  }
  .hero h1 {
    font-size: 24px !important;
  }
  .results-heading {
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 12px !important;
  }
  .tab-nav button, .tabs button, button[role="tab"] {
    padding: 8px 12px !important;
    font-size: 13px !important;
  }
  .upload-panel {
    padding: 16px !important;
  }
  .final-output-card {
    padding: 16px !important;
  }
}
"""


# Escapes HTML special characters in arbitrary text for safe browser rendering.
def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


# Converts a duration in seconds into a formatted HH:MM:SS or MM:SS timestamp string for the UI.
def _format_time(seconds: Any) -> str:
    try:
        value = max(0, float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


# Formats timestamped transcript segments for display inside the UI transcript tabs.
def _transcript_view(segments: list[dict[str, Any]], fallback: str) -> str:
    if not segments:
        return fallback
    return "\n".join(
        f"[{_format_time(segment.get('start'))} – {_format_time(segment.get('end'))}] "
        f"{segment.get('text', '').strip()}"
        for segment in segments
        if segment.get("text", "").strip()
    )


# Renders the HTML stepper showing progress through the 10 pipeline stages.
def _status_view(active_stage: str, error: bool = False) -> str:
    active_index = next(
        (index for index, (stage_id, _, _) in enumerate(STAGES) if stage_id == active_stage),
        0,
    )
    title = (
        "Processing complete"
        if active_stage == "complete"
        else "Processing stopped" if error else "Processing meeting"
    )
    current_copy = (
        "The pipeline stopped at this stage. Review the error details below."
        if error
        else STAGE_DETAILS.get(active_stage, "Working on your meeting.")
    )
    stages = []
    for index, (_, label, _) in enumerate(STAGES):
        if index < active_index or active_stage == "complete":
            state = "done"
            mark = "✓"
        elif index == active_index:
            state = "active"
            mark = "•"
        else:
            state = "pending"
            mark = str(index + 1)
        stages.append(
            f'<div class="stage {state}"><span class="stage-mark" aria-hidden="true">'
            f"{mark}</span><span>{_escape(label)}</span></div>"
        )
    return (
        '<section class="progress-panel" role="status" aria-live="polite">'
        f'<div class="progress-heading">{_escape(title)}</div>'
        f'<div class="progress-copy">{_escape(current_copy)}</div>'
        f'<div class="stage-list">{"".join(stages)}</div>'
        "</section>"
    )


# Renders a user-friendly error card with optional collapsible technical details.
def _error_view(result: dict[str, Any]) -> str:
    message = result.get("friendly_error", "We couldn't process this recording.")
    technical = result.get("technical_error", "")
    details = (
        f"<details><summary>Technical details</summary><pre>{_escape(technical)}</pre></details>"
        if technical
        else ""
    )
    return (
        '<section class="error-panel" role="alert">'
        '<div class="error-title">Something went wrong</div>'
        f'<div class="error-message">{_escape(message)}</div>{details}</section>'
    )


# Translates raw backend exceptions into actionable, user-friendly messages based on stage.
def _friendly_error(exc: Exception, stage: str) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if "unsupported audio" in lowered:
        return "This audio format is not supported. Upload a WAV, MP3, M4A, MP4, MPEG, MPGA, FLAC, or WebM file."
    if "empty" in lowered:
        return "The uploaded recording is empty. Choose a non-empty audio file and try again."
    if "corrupt" in lowered or "cannot be read" in lowered or "unreadable" in lowered:
        return "The uploaded audio file is corrupt or unreadable. Please provide a valid, playable audio recording."
    if "silent" in lowered:
        return "The uploaded audio file is completely silent. Please provide a recording with audible speech."
    if "duration" in lowered:
        return "Audio duration is outside supported limits. Please provide a recording between 0.5 seconds and 4 hours."
    if "not a file" in lowered or "missing" in lowered:
        return "We couldn't find the uploaded audio file. Re-upload the recording and try again."
    stage_messages = {
        "transcribing": "Local transcription failed. Check that the audio is valid and playable, then try again.",
        "refining": "Refinement failed. Raw transcript is available, but the refined transcript and final meeting record could not be generated.",
        "documenting": "Meeting record generation failed. Please try again.",
        "validating": "Evidence validation failed. Please try again.",
        "exporting": "The meeting was processed, but its download files could not be prepared.",
    }
    return stage_messages.get(stage, "We couldn't process this recording. Please try again.")


# Maps pipeline execution results into UI component values, visibility toggles, and downloads.
def _result_component_values(result: dict[str, Any] | None) -> tuple[Any, ...]:
    if not result or not isinstance(result, dict) or result.get("error"):
        partial = result if isinstance(result, dict) else {}
        has_partial = bool(
            partial.get("raw_transcript") or partial.get("refined_transcript")
        )
        return (
            _error_view(partial),
            gr.update(visible=not has_partial),
            gr.update(visible=has_partial),
            gr.update(interactive=True),
            "## Partial results" if has_partial else "",
            _transcript_view(
                partial.get("raw_segments", []), partial.get("raw_transcript", "")
            ),
            _transcript_view(
                partial.get("refined_segments", []),
                partial.get("refined_transcript", ""),
            ),
            "*Meeting record could not be generated due to an error.*",
            None,
        )

    record = result.get("record") or {}
    record_md = result.get("record_markdown") or render_markdown(record)
    archive_path = result.get("archive_path")
    archive_file = (
        str(archive_path)
        if archive_path and Path(archive_path).is_file()
        else None
    )

    warning_banner = ""
    if result.get("refinement_warning"):
        warning_banner = (
            '<section class="warning-panel" role="alert" style="background:#FFFBEB; '
            'border:1px solid #F59E0B; border-radius:8px; padding:12px 16px; margin-bottom:16px; color:#92400E;">'
            f'<strong>⚠️ Refinement Notice:</strong> {_escape(str(result["refinement_warning"]))}'
            '</section>'
        )

    return (
        warning_banner,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=True),
        f"## {_escape(record.get('meeting_title') or 'Meeting Results')}",
        _transcript_view(result.get("raw_segments", []), result.get("raw_transcript", "")),
        _transcript_view(
            result.get("refined_segments", []), result.get("refined_transcript", "")
        ),
        record_md,
        archive_file,
    )


# Runs the meeting pipeline in a background thread while yielding stage updates to the UI.
def _run_for_ui(audio_path: str | None, glossary: str):
    if not audio_path:
        result = {
            "error": True,
            "stage": "received",
            "friendly_error": "Upload a meeting recording before starting processing.",
            "technical_error": "No audio filepath was provided by the Gradio upload component.",
        }
        yield _status_view("received"), result, gr.update(interactive=True)
        return

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    active_stage = {"value": "received"}
    audio_file = Path(audio_path)
    partial_result: dict[str, Any] = {}

    # Pushes pipeline stage transitions into the UI event queue.
    def notify_stage(stage: str) -> None:
        active_stage["value"] = stage
        events.put(("stage", stage))

    # Stores intermediate transcripts to display partial progress if later stages fail.
    def capture_partial(values: dict[str, Any]) -> None:
        partial_result.update(values)

    # Worker function executing the processing pipeline and posting final results.
    def process() -> None:
        try:
            result = process_meeting(
                audio_file,
                glossary,
                stage_callback=notify_stage,
                partial_result_callback=capture_partial,
            )
            result["output_dir"] = str(result["output_dir"])
            result["archive_path"] = str(result["archive_path"])
            events.put(("result", result))
        except Exception as exc:
            logger.exception("Meeting processing failed during %s", active_stage["value"])
            events.put(
                (
                    "result",
                    {
                        "error": True,
                        "stage": active_stage["value"],
                        "friendly_error": _friendly_error(exc, active_stage["value"]),
                        "technical_error": f"{type(exc).__name__}: {sanitize_exception(exc)}",
                        **partial_result,
                    },
                )
            )

    logger.info("Starting UI processing for uploaded audio: %s", audio_file.name)
    worker = threading.Thread(target=process, name="meeting-processing", daemon=False)
    yield _status_view("received"), None, gr.update(interactive=False)
    worker.start()
    while True:
        event_type, payload = events.get()
        if event_type == "stage":
            yield _status_view(payload), None, gr.update(interactive=False)
            continue
        final_stage = payload.get("stage", "complete")
        yield (
            _status_view(final_stage, error=bool(payload.get("error"))),
            payload,
            gr.update(interactive=True),
        )
        return


THEME = (
    gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="sm",
        spacing_size="sm",
    )
    .set(
        body_background_fill="#FFFFFF",
        body_background_fill_dark="#FFFFFF",
        body_text_color="#111827",
        body_text_color_dark="#111827",
        body_text_color_subdued="#4B5563",
        body_text_color_subdued_dark="#4B5563",
        background_fill_primary="#FFFFFF",
        background_fill_primary_dark="#FFFFFF",
        background_fill_secondary="#F9FAFB",
        background_fill_secondary_dark="#F9FAFB",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        block_label_text_color="#111827",
        block_label_text_color_dark="#111827",
        block_title_text_color="#111827",
        block_title_text_color_dark="#111827",
        border_color_primary="#D1D5DB",
        border_color_primary_dark="#D1D5DB",
    )
)

with gr.Blocks(title="AI Meeting Assistant", fill_width=True) as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="app-header"):
            gr.HTML(
                '<div class="header-container">'
                '<div class="header-left">'
                '  <div class="header-brand-row">'
                '    <div id="brand-mark" aria-hidden="true">A</div>'
                '    <p class="brand-statement">Turn meeting recordings into structured, actionable records.</p>'
                '  </div>'
                '  <div class="header-workflow" aria-label="Meeting workflow steps">'
                '    <div class="workflow-step">'
                '      <svg class="workflow-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>'
                '      <span>Audio</span>'
                '    </div>'
                '    <span class="workflow-arrow" aria-hidden="true">→</span>'
                '    <div class="workflow-step">'
                '      <svg class="workflow-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>'
                '      <span>Refine</span>'
                '    </div>'
                '    <span class="workflow-arrow" aria-hidden="true">→</span>'
                '    <div class="workflow-step">'
                '      <svg class="workflow-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/></svg>'
                '      <span>Meeting Record</span>'
                '    </div>'
                '  </div>'
                '</div>'
                '<div class="header-right">'
                '  <div class="brand-title">AI Meeting Assistant</div>'
                '  <div class="brand-subtitle">Audio transcription, refinement &amp; documentation</div>'
                '</div>'
                '</div>'
            )

        status = gr.HTML(value="", elem_id="processing-status")
        result_state = gr.State()
        error_message = gr.HTML(value="")

        with gr.Column(visible=True, elem_id="landing-view") as landing_view:
            gr.HTML(
                '<section class="hero"><h1>Meeting Intelligence Workspace</h1>'
                "<p>Upload meeting audio to generate a raw transcript, a domain-refined transcript, and structured meeting documentation.</p></section>"
            )
            with gr.Column(elem_classes="upload-panel"):
                gr.Markdown("### Upload meeting audio\nYour recording is transcribed locally with faster-whisper.")
                audio = gr.Audio(
                    type="filepath",
                    sources=["upload"],
                    label="Choose or drop an audio recording",
                    elem_classes="upload-audio",
                )
                formats = " · ".join(
                    extension.lstrip(".").upper()
                    for extension in sorted(SUPPORTED_AUDIO_EXTENSIONS)
                )
                gr.Markdown(
                    f"<div style='color: #4B5563; font-size: 12px; margin-top: 4px;'>Supported formats: {formats}</div>"
                )
                glossary = gr.Textbox(
                    label="Optional terminology glossary",
                    placeholder="Names, acronyms, or technical terms separated by commas",
                    lines=1,
                )
                submit = gr.Button(
                    "Start Processing",
                    variant="primary",
                    elem_classes="primary-cta",
                )

        with gr.Column(visible=False, elem_id="results-view") as results_view:
            with gr.Row(elem_classes="results-heading"):
                result_title = gr.Markdown("## Meeting Results")
                with gr.Row():
                    zip_download = gr.DownloadButton(
                        label="Download Results Package",
                        value=None,
                        variant="primary",
                        size="sm",
                    )
                    another_meeting = gr.Button(
                        "Process Another Meeting",
                        variant="secondary",
                        size="sm",
                    )
            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("Raw Transcript", id="tab-raw"):
                    raw_transcript = gr.Textbox(
                        label="Raw Transcript (Local Whisper)",
                        lines=22,
                        max_lines=32,
                        buttons=["copy"],
                        interactive=False,
                        elem_classes="transcript-box",
                    )
                with gr.Tab("Refined Transcript", id="tab-refined"):
                    refined_transcript = gr.Textbox(
                        label="Refined Transcript (Domain-Corrected)",
                        lines=22,
                        max_lines=32,
                        buttons=["copy"],
                        interactive=False,
                        elem_classes="transcript-box",
                    )
                with gr.Tab("Final Output", id="tab-final"):
                    final_output = gr.Markdown(
                        elem_classes="final-output-card",
                    )

    submit.click(
        _run_for_ui,
        inputs=[audio, glossary],
        outputs=[status, result_state, submit],
        show_progress="hidden",
    ).then(
        _result_component_values,
        inputs=[result_state],
        outputs=[
            error_message,
            landing_view,
            results_view,
            submit,
            result_title,
            raw_transcript,
            refined_transcript,
            final_output,
            zip_download,
        ],
        show_progress="hidden",
    )

    another_meeting.click(
        lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            gr.update(interactive=True),
            None,
            "",
        ),
        outputs=[
            landing_view,
            results_view,
            error_message,
            status,
            submit,
            audio,
            glossary,
        ],
        show_progress="hidden",
    )


# Launches the Gradio web application with custom CSS styling and light theme configuration.
def launch_app() -> tuple[Any, str, str]:
    return demo.launch(theme=THEME, css=CSS)
