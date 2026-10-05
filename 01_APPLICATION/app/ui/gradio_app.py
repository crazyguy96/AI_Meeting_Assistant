import html
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from app.main import process_meeting
from app.pipeline.transcription import SUPPORTED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

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
OUTPUT_FILES = (
    ("Meeting record · JSON", "meeting_record.json"),
    ("Meeting record · Markdown", "meeting_record.md"),
    ("Raw transcript", "raw_transcript.txt"),
    ("Refined transcript", "refined_transcript.txt"),
    ("Decisions", "decisions.json"),
    ("Action items", "action_items.json"),
    ("Minutes", "minutes.md"),
)

CSS = """
:root {
  color-scheme: light !important;
  --page: #f5f7fa;
  --surface: #ffffff;
  --ink: #17212f;
  --muted: #64748b;
  --line: #e4e9f0;
  --accent: #315efb;
  --accent-dark: #2449d8;
  --success: #16845b;
  --warning: #9a5c08;
  --danger: #a83232;
}
html, body, gradio-app, gradio-app > .main,
.gradio-container, .gradio-container .main {
  color-scheme: light !important;
  background-color: #f5f7fa !important;
}
body, .gradio-container {
  background: var(--page) !important;
  color: var(--ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: 1120px !important;
  width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
  margin: 0 auto !important;
}
gradio-app > .main.fillable {
  width: 100% !important;
  box-sizing: border-box !important;
  padding-left: clamp(10px, 3vw, 32px) !important;
  padding-right: clamp(10px, 3vw, 32px) !important;
}
gradio-app .gradio-container .wrap,
gradio-app .gradio-container main.contain {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
}
#app-shell .prose,
#app-shell .prose :where(h1, h2, h3, h4, p, li, strong) {
  color: var(--ink) !important;
}
#app-shell { width: 100%; min-width: 0 !important; gap: 22px; padding: 24px 18px 42px; }
#app-header {
  align-items: center;
  width: 100%;
  min-width: 0 !important;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--line);
  padding: 6px 0 18px;
}
#app-header > div { min-width: 0 !important; }
#app-header > .row { flex: 1 1 280px; }
#app-header > .row > .block:first-child { flex: 0 0 36px; width: 36px; min-width: 36px; }
#app-header > .row > .block:last-child { flex: 1 1 200px; min-width: 200px; }
#app-header > .block { flex: 0 0 auto; width: auto; }
#brand-mark {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: #eaf0ff; color: var(--accent); font-weight: 750;
  border: 1px solid #d7e1ff;
}
.brand-title { color: var(--ink) !important; font-size: 17px; font-weight: 680; letter-spacing: -0.025em; }
.brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
.ready-pill {
  display: inline-flex; gap: 8px; align-items: center;
  padding: 7px 11px; color: #17684b; background: #edf8f2;
  border: 1px solid #d9eee2; border-radius: 999px;
  font-size: 12px; font-weight: 600;
}
.ready-dot { width: 7px; height: 7px; border-radius: 50%; background: #1e9a68; }
.hero { text-align: center; padding: 32px 12px 16px; }
.hero h1 { color: var(--ink) !important; font-size: clamp(30px, 5vw, 43px); line-height: 1.13; letter-spacing: -.045em; margin: 0 0 12px; }
.hero p { color: var(--muted); font-size: 16px; line-height: 1.6; max-width: 620px; margin: 0 auto; }
.feature-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 22px auto 24px; }
.feature-chip {
  padding: 7px 11px; font-size: 12px; font-weight: 560;
  color: #46556a; border: 1px solid var(--line); background: #fff; border-radius: 7px;
}
.panel, .result-card, .upload-panel {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 11px; box-shadow: 0 2px 8px rgba(20, 35, 60, .035);
}
.upload-panel { padding: 22px; max-width: 760px; margin: 0 auto; }
.upload-panel h2 { font-size: 17px; margin: 0 0 5px; letter-spacing: -.02em; }
.upload-note, .privacy-note { color: var(--muted); font-size: 12px; line-height: 1.55; }
.privacy-note { text-align: center; margin: 14px auto 0; max-width: 650px; }
.upload-audio > div { border: 1px dashed #b8c5d7 !important; border-radius: 9px !important; background: #fbfcfe !important; }
.upload-audio label { color: var(--ink) !important; }
.glossary label { color: #344256 !important; font-size: 13px !important; }
.primary-cta button {
  background: var(--accent) !important; border: 1px solid var(--accent) !important;
  border-radius: 8px !important; min-height: 46px !important;
  font-weight: 650 !important; box-shadow: 0 2px 5px rgba(49, 94, 251, .18);
}
.primary-cta button:hover { background: var(--accent-dark) !important; }
.progress-panel, .error-panel, .success-panel {
  background: #fff; border: 1px solid var(--line); border-radius: 10px;
  padding: 17px 19px; margin: 2px 0 4px;
}
.progress-heading { color: var(--ink); font-size: 15px; font-weight: 680; margin-bottom: 4px; }
.progress-copy { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
.stage-list { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 9px; }
.stage {
  display: flex; gap: 8px; align-items: flex-start; min-height: 42px;
  padding: 9px; border-radius: 7px; background: #f7f8fa;
  color: #7b8798; font-size: 11px; line-height: 1.35;
}
.stage-mark {
  flex: 0 0 17px; width: 17px; height: 17px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid #cbd3df; color: #78869a; font-size: 10px; font-weight: 700;
}
.stage.done { color: #28684d; background: #f2f9f5; }
.stage.done .stage-mark { border-color: #8bc9a9; color: #16845b; background: #fff; }
.stage.active { color: #2449b5; border: 1px solid #c9d5ff; background: #f3f6ff; font-weight: 630; }
.stage.active .stage-mark { border-color: var(--accent); color: #fff; background: var(--accent); }
.error-panel { border-color: #f0caca; background: #fffafa; color: var(--danger); }
.error-title { font-weight: 680; font-size: 15px; margin-bottom: 4px; }
.error-message { color: #633b3b; font-size: 13px; line-height: 1.5; }
.error-panel details { color: #6b7280; font-size: 11px; margin-top: 11px; }
.error-panel pre { overflow: auto; white-space: pre-wrap; color: #475569; }
.results-heading { border-bottom: 1px solid var(--line); padding: 4px 0 16px; }
.results-heading h1 { color: var(--ink) !important; font-size: 25px; letter-spacing: -.035em; margin: 0 0 4px; }
.results-heading p { color: var(--muted); margin: 0; font-size: 13px; }
.result-card { padding: 18px; }
.stat-card { padding: 13px 15px; border: 1px solid var(--line); background: #fff; border-radius: 9px; }
.stat-number { color: var(--ink); font-size: 22px; font-weight: 700; letter-spacing: -.04em; }
.stat-label { color: var(--muted); font-size: 11px; margin-top: 2px; }
.stats-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; width: min(100%, 420px); }
.overview-summary { font-size: 14px; line-height: 1.7; color: #38475b; white-space: pre-wrap; }
.record-heading { color: var(--ink); font-size: 15px; font-weight: 670; margin: 0 0 11px; }
.decision-card, .task-card, .proposal-card {
  border: 1px solid var(--line); border-radius: 9px; padding: 15px 16px;
  background: #fff; margin: 0 0 11px;
}
.decision-card { border-left: 3px solid #23956a; }
.proposal-card { border-left: 3px solid #d69a35; }
.task-card { border-left: 3px solid var(--accent); }
.item-eyebrow { font-size: 10px; letter-spacing: .08em; font-weight: 720; color: var(--muted); text-transform: uppercase; margin-bottom: 7px; }
.decision-card .item-eyebrow { color: #19724f; }
.proposal-card .item-eyebrow { color: var(--warning); }
.item-title { color: var(--ink); font-size: 14px; font-weight: 620; line-height: 1.5; }
.item-status { color: var(--muted); font-size: 11px; margin-top: 6px; }
.item-evidence { border-top: 1px solid #edf0f4; color: #58677b; font-size: 12px; line-height: 1.55; margin-top: 12px; padding-top: 10px; }
.task-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 13px; }
.meta-label { color: var(--muted); display: block; font-size: 10px; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 3px; }
.meta-value { color: #29384d; font-size: 12px; font-weight: 580; overflow-wrap: anywhere; }
.empty-state { color: var(--muted); font-size: 13px; padding: 18px 0; }
.minutes-list { color: #38475b; font-size: 13px; line-height: 1.7; padding-left: 19px; }
.transcript-hint { color: var(--muted); font-size: 12px; line-height: 1.5; margin-bottom: 10px; }
.transcript-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 12px !important; line-height: 1.75 !important; }
.download-row { align-items: center; padding: 10px 0; border-bottom: 1px solid #edf0f4; }
.download-label { font-size: 13px; color: #36455a; font-weight: 550; }
.download-row button { min-height: 34px !important; border-radius: 7px !important; font-size: 12px !important; }
.tabs { border: 0 !important; }
.tab-nav button { font-size: 13px !important; }
.footer-note { text-align: center; color: #8490a1; font-size: 11px; padding-top: 6px; }
@media (max-width: 700px) {
  #app-shell { padding: 14px 10px 28px; gap: 15px; }
  #app-header > .row { flex-basis: 280px; }
  #app-header > .block { margin-left: auto; }
  .hero { padding-top: 23px; }
  .upload-panel { padding: 15px; }
  .stage-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .progress-panel { padding: 14px; }
  .results-heading h1 { font-size: 22px; }
}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _format_time(seconds: Any) -> str:
    try:
        value = max(0, float(seconds))
    except (TypeError, ValueError):
        return "--:--"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def _transcript_view(segments: list[dict[str, Any]], fallback: str) -> str:
    if not segments:
        return fallback
    return "\n".join(
        f"[{_format_time(segment.get('start'))} – {_format_time(segment.get('end'))}] "
        f"{segment.get('text', '')}"
        for segment in segments
    )


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
        f'<div class="progress-copy">{_escape(current_copy)} '
        "Longer recordings may take a few minutes.</div>"
        f'<div class="stage-list">{"".join(stages)}</div>'
        "</section>"
    )


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


def _friendly_error(exc: Exception, stage: str) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    if "unsupported audio" in lowered:
        return "This audio format is not supported. Upload a WAV, MP3, M4A, MP4, MPEG, MPGA, FLAC, or WebM file."
    if "empty" in lowered:
        return "The uploaded recording is empty. Choose a non-empty audio file and try again."
    if "cannot be read" in lowered or "not a file" in lowered or "missing" in lowered:
        return "We couldn't read the uploaded audio file. Re-upload the recording and try again."
    stage_messages = {
        "transcribing": "Local transcription failed. Check that the audio is valid and playable, then try again.",
        "refining": "Refinement failed. The raw transcript is available, but no refined transcript or final meeting record was generated.",
        "documenting": "Meeting record generation failed. Please try again.",
        "validating": "Evidence validation failed. Please try again.",
        "exporting": "The meeting was processed, but its download files could not be prepared.",
    }
    return stage_messages.get(stage, "We couldn't process this recording. Please try again.")


def _render_decisions(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<div class="empty-state">No supported decisions were recorded.</div>'
    cards = []
    for item in items:
        status = item.get("status", "Decision")
        evidence = item.get("evidence_quote") or "No evidence quote provided."
        cards.append(
            '<article class="decision-card"><div class="item-eyebrow">'
            f"{_escape(status)}</div><div class=\"item-title\">{_escape(item.get('text', ''))}</div>"
            f'<div class="item-evidence"><strong>Evidence</strong><br>“{_escape(evidence)}”</div>'
            "</article>"
        )
    return "".join(cards)


def _render_proposals(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<div class="empty-state">No proposals or unresolved items were recorded.</div>'
    return "".join(
        '<article class="proposal-card"><div class="item-eyebrow">'
        "Proposal · not decided</div>"
        f'<div class="item-title">{_escape(item.get("text", ""))}</div>'
        '<div class="item-status">No final decision recorded.</div>'
        + (
            f'<div class="item-evidence"><strong>Evidence</strong><br>“{_escape(item.get("evidence_quote", ""))}”</div>'
            if item.get("evidence_quote")
            else ""
        )
        + "</article>"
        for item in items
    )


def _render_actions(items: list[dict[str, Any]]) -> str:
    if not items:
        return '<div class="empty-state">No action items were recorded.</div>'
    cards = []
    for item in items:
        evidence = item.get("evidence_quote") or "No evidence quote provided."
        cards.append(
            '<article class="task-card"><div class="item-eyebrow">Action item</div>'
            f'<div class="item-title">{_escape(item.get("task", ""))}</div>'
            f'<div class="item-status">{_escape(item.get("status", "Needs Review"))}</div>'
            '<div class="task-meta">'
            f'<div><span class="meta-label">Owner</span><span class="meta-value">{_escape(item.get("owner") or "Not specified")}</span></div>'
            f'<div><span class="meta-label">Deadline</span><span class="meta-value">{_escape(item.get("deadline") or "Not specified")}</span></div>'
            "</div>"
            f'<div class="item-evidence"><strong>Evidence</strong><br>“{_escape(evidence)}”</div>'
            "</article>"
        )
    return "".join(cards)


def _render_overview(record: dict[str, Any]) -> str:
    summary = record.get("summary") or "No summary was generated."
    return (
        '<section class="result-card"><h2 class="record-heading">Meeting summary</h2>'
        f'<div class="overview-summary">{_escape(summary)}</div></section>'
    )


def _render_minutes(minutes: list[Any]) -> str:
    if not minutes:
        return '<div class="empty-state">No minutes were recorded.</div>'
    return (
        '<section class="result-card"><h2 class="record-heading">Meeting minutes</h2><ul class="minutes-list">'
        + "".join(f"<li>{_escape(item)}</li>" for item in minutes)
        + "</ul></section>"
    )


def _render_stats(record: dict[str, Any]) -> str:
    decisions = len(record.get("decisions", []))
    actions = len(record.get("action_items", []))
    return (
        '<div class="stats-grid"><div class="stat-card"><div class="stat-number">'
        f"{decisions}</div><div class=\"stat-label\">Decisions</div></div>"
        '<div class="stat-card"><div class="stat-number">'
        f"{actions}</div><div class=\"stat-label\">Action items</div></div></div>"
    )


def _result_component_values(result: dict[str, Any]) -> tuple[Any, ...]:
    if result.get("error"):
        partial = result
        has_partial_transcript = bool(
            partial.get("raw_transcript") or partial.get("refined_transcript")
        )
        return (
            _error_view(result),
            gr.update(visible=not has_partial_transcript),
            gr.update(visible=has_partial_transcript),
            gr.update(interactive=True),
            "## Partial results" if has_partial_transcript else "",
            "",
            "",
            '<div class="empty-state">No results to display yet.</div>',
            "",
            "",
            "",
            _transcript_view(
                partial.get("raw_segments", []), partial.get("raw_transcript", "")
            ),
            _transcript_view(
                partial.get("refined_segments", []),
                partial.get("refined_transcript", ""),
            ),
            '<div class="empty-state">The meeting record was not completed.</div>',
            '<div class="empty-state">The meeting record was not completed.</div>',
            "",
            "",
            None,
            *[None for _ in OUTPUT_FILES],
        )

    record = result["record"]
    output_dir = Path(result["output_dir"])
    downloads = [
        str(output_dir / filename) if (output_dir / filename).is_file() else None
        for _, filename in OUTPUT_FILES
    ]
    return (
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=True),
        f"## {_escape(record.get('meeting_title') or 'Meeting results')}",
        _render_stats(record),
        _render_overview(record),
        _render_decisions(record.get("decisions", [])),
        _render_proposals(record.get("non_decisions", [])),
        _render_actions(record.get("action_items", [])),
        _render_minutes(record.get("minutes", [])),
        _transcript_view(result.get("raw_segments", []), result.get("raw_transcript", "")),
        _transcript_view(
            result.get("refined_segments", []), result.get("refined_transcript", "")
        ),
        _render_decisions(record.get("decisions", [])),
        _render_actions(record.get("action_items", [])),
        json.dumps(record, ensure_ascii=False, indent=2),
        result.get("record_markdown", ""),
        str(result["archive_path"]) if Path(result["archive_path"]).is_file() else None,
        *downloads,
    )


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

    def notify_stage(stage: str) -> None:
        active_stage["value"] = stage
        events.put(("stage", stage))

    def capture_partial(values: dict[str, Any]) -> None:
        partial_result.update(values)

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
                        "technical_error": f"{type(exc).__name__}: {exc}",
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


def _download_row(label: str, filename: str) -> None:
    with gr.Row(elem_classes="download-row"):
        gr.Markdown(f'<div class="download-label">{_escape(label)}</div>')
        download_button = gr.DownloadButton(
            label="Download",
            value=None,
            size="sm",
            interactive=True,
        )
        DOWNLOAD_BUTTONS.append(download_button)


DOWNLOAD_BUTTONS: list[gr.DownloadButton] = []

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="sm",
    spacing_size="sm",
)

with gr.Blocks(title="AI Meeting Assistant", fill_width=True) as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="app-header"):
            with gr.Row():
                gr.HTML('<div id="brand-mark" aria-hidden="true">A</div>')
                gr.HTML(
                    '<div><div class="brand-title">AI Meeting Assistant</div>'
                    '<div class="brand-subtitle">Meeting intelligence workspace</div></div>'
                )
            gr.HTML(
                '<div class="ready-pill"><span class="ready-dot"></span>'
                "System ready</div>"
            )

        status = gr.HTML(value="", elem_id="processing-status")
        result_state = gr.State()
        error_message = gr.HTML(value="")

        with gr.Column(visible=True, elem_id="landing-view") as landing_view:
            gr.HTML(
                '<section class="hero"><h1>From meeting audio to accurate, '
                "actionable records.</h1>"
                "<p>Move from spoken discussion to a clear, evidence-grounded "
                "record your team can act on.</p></section>"
                '<div class="feature-row"><span class="feature-chip">Local Whisper transcription</span>'
                '<span class="feature-chip">Terminology refinement</span>'
                '<span class="feature-chip">Decisions &amp; action items</span></div>'
            )
            with gr.Column(elem_classes="upload-panel"):
                gr.Markdown("### Upload meeting audio\nYour recording stays on this device for transcription.")
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
                    f'<div class="upload-note">Supported formats: {formats}</div>'
                )
                glossary = gr.Textbox(
                    label="Optional terminology",
                    placeholder="Add names, acronyms, or technical terms separated by commas",
                    lines=1,
                    elem_classes="glossary",
                )
                submit = gr.Button(
                    "Start processing",
                    variant="primary",
                    elem_classes="primary-cta",
                )
            gr.Markdown(
                "Audio is transcribed locally with Whisper, then refined and documented "
                "using domain-aware language models.",
                elem_classes="privacy-note",
            )

        with gr.Column(visible=False, elem_id="results-view") as results_view:
            with gr.Row(elem_classes="results-heading"):
                with gr.Column():
                    result_title = gr.Markdown("## Meeting results")
                    gr.Markdown("Review the generated summary, evidence, and downloadable files.")
                zip_download = gr.DownloadButton(
                    label="Download complete results",
                    value=None,
                    variant="primary",
                    size="sm",
                )
                another_meeting = gr.Button(
                    "Process another meeting",
                    variant="secondary",
                    size="sm",
                )
            with gr.Row():
                stats = gr.HTML()
            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("Overview"):
                    overview = gr.HTML()
                    gr.Markdown("### Key decisions")
                    overview_decisions = gr.HTML()
                    gr.Markdown("### Action items")
                    overview_actions = gr.HTML()
                with gr.Tab("Transcript"):
                    gr.Markdown(
                        '<div class="transcript-hint">The original Whisper output is '
                        "preserved separately from the refined transcript. Timestamps "
                        "come from Whisper's audio segments.</div>"
                    )
                    with gr.Tabs():
                        with gr.Tab("Raw transcript"):
                            raw_transcript = gr.Textbox(
                                label="Raw transcript · local Whisper",
                                lines=20,
                                max_lines=24,
                                buttons=["copy"],
                                interactive=False,
                                elem_classes="transcript-box",
                            )
                        with gr.Tab("Refined transcript"):
                            refined_transcript = gr.Textbox(
                                label="Refined transcript · terminology checked",
                                lines=20,
                                max_lines=24,
                                buttons=["copy"],
                                interactive=False,
                                elem_classes="transcript-box",
                            )
                with gr.Tab("Decisions"):
                    gr.Markdown("### Confirmed decisions")
                    decisions = gr.HTML()
                    gr.Markdown("### Discussed, not decided")
                    proposals = gr.HTML()
                with gr.Tab("Action items"):
                    gr.Markdown("### Assigned work")
                    actions = gr.HTML()
                with gr.Tab("Minutes"):
                    minutes = gr.HTML()
                with gr.Tab("Meeting record"):
                    gr.Markdown(
                        "### Final validated record\n"
                        "This structured record is separate from both transcript stages."
                    )
                    meeting_record_json = gr.Textbox(
                        label="Machine-readable meeting record · JSON",
                        lines=20,
                        max_lines=32,
                        buttons=["copy"],
                        interactive=False,
                        elem_classes="transcript-box",
                    )
                    meeting_record_markdown = gr.Markdown()
                with gr.Tab("Export"):
                    gr.Markdown("### Download meeting outputs\nOnly files created for this run are available.")
                    with gr.Column(elem_classes="result-card"):
                        for label, filename in OUTPUT_FILES:
                            _download_row(label, filename)
            gr.Markdown(
                "Evidence, owners, and deadlines are displayed exactly as returned by the validated record.",
                elem_classes="footer-note",
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
            stats,
            overview,
            overview_decisions,
            proposals,
            overview_actions,
            minutes,
            raw_transcript,
            refined_transcript,
            decisions,
            actions,
            meeting_record_json,
            meeting_record_markdown,
            zip_download,
            *DOWNLOAD_BUTTONS,
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


def launch_app() -> tuple[Any, str, str]:
    return demo.launch(theme=THEME, css=CSS)
