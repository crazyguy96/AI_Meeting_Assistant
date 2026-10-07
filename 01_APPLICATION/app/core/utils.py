import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import OUTPUT_DIR


# Redacts sensitive Groq API keys and Bearer tokens from exception messages before logging or UI display.
def sanitize_exception(exc: Any) -> str:
    msg = str(exc)
    msg = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", msg)
    msg = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-]+", r"\1[REDACTED_API_KEY]", msg)
    return msg


# Formats the structured meeting record dictionary into clean, human-readable Markdown.
def render_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# {record.get('meeting_title') or 'Meeting'}",
        "",
        "## Summary",
        record.get("summary") or "No summary generated.",
        "",
        "## Minutes",
    ]
    lines.extend(f"- {item}" for item in record.get("minutes", []))
    lines.extend(["", "## Decisions"])
    lines.extend(
        f"- {item.get('text', '')} — {item.get('status', 'Needs Review')} "
        f"| Evidence: {item.get('evidence_quote', 'Unspecified')}"
        for item in record.get("decisions", [])
    )
    lines.extend(["", "## Discussed, Not Decided"])
    lines.extend(
        f"- {item.get('text', '')}" for item in record.get("non_decisions", [])
    )
    lines.extend(["", "## Action Items"])
    for item in record.get("action_items", []):
        task = item.get("task", "")
        owner = item.get("owner", "Unspecified")
        deadline = item.get("deadline", "Unspecified")
        status = item.get("status", "Needs Review")
        evidence = item.get("evidence_quote", "Unspecified")
        action_line = (
            f"- {task} — Owner: {owner}; Deadline: {deadline}; "
            f"Status: {status} | Evidence: {evidence}"
        )
        if item.get("owner_note"):
            action_line += f" ({item['owner_note']})"
        lines.append(action_line)
    for key, heading in (
        ("discussion_points", "Discussion Points"),
        ("open_questions", "Open Questions"),
    ):
        lines.extend(["", f"## {heading}"])
        lines.extend(f"- {item}" for item in record.get(key, []))
    return "\n".join(lines).rstrip() + "\n"


# Persists all meeting outputs and audio to a timestamped folder and creates a downloadable zip archive.
def save_outputs(
    transcription: dict[str, Any],
    refined: dict[str, Any],
    record: dict[str, Any],
    audio_path: str | Path,
) -> tuple[Path, Path]:
    run_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "raw_transcript.txt").write_text(
        transcription["text"], encoding="utf-8"
    )
    (run_dir / "refined_transcript.txt").write_text(
        refined["refined_text"], encoding="utf-8"
    )
    (run_dir / "minutes.md").write_text(
        "\n".join(
            ["# Meeting Minutes", ""]
            + [f"- {item}" for item in record.get("minutes", [])]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "decisions.json").write_text(
        json.dumps(record.get("decisions", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "action_items.json").write_text(
        json.dumps(record.get("action_items", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "meeting_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "meeting_record.md").write_text(
        render_markdown(record), encoding="utf-8"
    )
    audit = {
        key: (
            [
                e
                for e in refined.get(key, [])
                if not (isinstance(e, dict) and str(e.get("from", "")).strip() == str(e.get("to", "")).strip())
            ]
            if key in ("proposed_edits", "applied_edits", "rejected_edits")
            else refined.get(key, [])
        )
        for key in ("glossary", "proposed_edits", "applied_edits", "rejected_edits")
    }
    (run_dir / "refinement_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source_audio = Path(audio_path)
    shutil.copy2(source_audio, run_dir / source_audio.name)
    archive_path = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return run_dir, archive_path
