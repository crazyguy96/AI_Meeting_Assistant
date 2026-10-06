import logging
from pathlib import Path
from typing import Any, Callable

from app.core.utils import render_markdown, sanitize_exception, save_outputs
from app.pipeline.documentation import generate_record
from app.pipeline.evidence import validate_record
from app.pipeline.refinement import refine_transcription
from app.pipeline.transcription import transcribe_audio, validate_audio_file

logger = logging.getLogger(__name__)

# Coordinates the full end-to-end meeting processing pipeline across all 10 stages:
# validates audio, runs Faster-Whisper, refines transcript with Groq, creates structured
# meeting documentation, validates evidence, and packages all output deliverables.
def process_meeting(
    audio_path: str | Path,
    glossary: str = "",
    stage_callback: Callable[[str], None] | None = None,
    partial_result_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not audio_path:
        raise ValueError("Please upload a meeting recording.")
    audio_file = Path(audio_path)
    logger.info("Received uploaded meeting audio: %s", audio_file.name)
    if stage_callback:
        stage_callback("received")

    validate_audio_file(audio_file)
    if stage_callback:
        stage_callback("validated")

    terms = [term.strip() for term in str(glossary).split(",") if term.strip()]
    if stage_callback:
        stage_callback("transcribing")
    raw = transcribe_audio(audio_file)
    if partial_result_callback:
        partial_result_callback(
            {
                "raw_transcript": raw["text"],
                "raw_segments": raw.get("segments", []),
            }
        )
    if stage_callback:
        stage_callback("raw_ready")
    logger.info("Starting transcript refinement")
    if stage_callback:
        stage_callback("refining")
    refined = refine_transcription(raw, terms)
    if partial_result_callback:
        partial_result_callback(
            {
                "raw_transcript": raw["text"],
                "raw_segments": raw.get("segments", []),
                "refined_transcript": refined["refined_text"],
                "refined_segments": refined.get("refined_segments", []),
            }
        )
    if stage_callback:
        stage_callback("refined")
    logger.info("Starting meeting documentation")
    if stage_callback:
        stage_callback("documenting")
    generated_record = generate_record(refined)
    if stage_callback:
        stage_callback("validating")
    print("[STAGE] evidence", flush=True)
    try:
        record = validate_record(generated_record, refined)
    except Exception as exc:
        print("[ERROR] stage=evidence", flush=True)
        print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
        raise
    if stage_callback:
        stage_callback("exporting")
    output_dir, archive_path = save_outputs(raw, refined, record, audio_file)
    if stage_callback:
        stage_callback("complete")
    return {
        "raw_transcript": raw["text"],
        "raw_segments": raw.get("segments", []),
        "refined_transcript": refined["refined_text"],
        "refined_segments": refined.get("refined_segments", []),
        "record": record,
        "record_markdown": render_markdown(record),
        "output_dir": output_dir,
        "archive_path": archive_path,
    }
