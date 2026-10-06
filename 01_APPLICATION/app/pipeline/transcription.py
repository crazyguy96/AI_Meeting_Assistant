from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from app.core.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

# Set of supported audio file formats accepted by the pipeline.
SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".wav",
    ".webm",
}


# Validates that the uploaded recording exists, has an allowed format, is non-empty, and is readable.
def validate_audio_file(file_path: str | Path) -> Path:
    path = Path(file_path)
    logger.info("Validating uploaded audio: %s", path.name)
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio file type. Supported extensions: {supported}.")
    if not path.is_file():
        raise ValueError("The uploaded recording is missing or is not a file.")
    try:
        if path.stat().st_size == 0:
            raise ValueError("The uploaded recording is empty.")
        with path.open("rb") as audio_file:
            audio_file.read(1)
    except OSError as exc:
        raise ValueError(f"The uploaded recording cannot be read: {path.name}") from exc
    logger.info(
        "Audio validation complete: %s (%d bytes)",
        path.name,
        path.stat().st_size,
    )
    return path


# Loads and caches the local Faster-Whisper model on CPU using int8 quantization.
@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    logger.info(
        "Loading faster-whisper model %s on %s with %s compute",
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
    )
    return WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


# Converts a duration in seconds into a formatted HH:MM:SS or MM:SS timestamp string.
def format_time(seconds: Any) -> str:
    try:
        value = max(0, float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    minutes, remainder = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


# Transcribes the validated audio file using Faster-Whisper into timestamped raw segments and full text.
def transcribe_audio(file_path: str | Path) -> dict[str, Any]:
    logger.info("Received uploaded audio for transcription: %s", Path(file_path).name)
    path = validate_audio_file(file_path)
    logger.info(
        "Loading/reusing faster-whisper model %s on %s (%s)",
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
    )
    model = get_whisper_model()
    logger.info("Starting English transcription: %s", path.name)
    segments_iterator, info = model.transcribe(
        str(path),
        language="en",
    )
    segments = [
        {
            "id": f"S{index:04d}",
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for index, segment in enumerate(segments_iterator, start=1)
    ]
    text = " ".join(segment["text"] for segment in segments).strip()
    if not text:
        raise RuntimeError("faster-whisper returned an empty transcript.")
    logger.info(
        "Finished transcription: %s (%d timestamped segments)",
        path.name,
        len(segments),
    )
    timestamped_lines = [
        f"[{format_time(s['start'])} – {format_time(s['end'])}] {s['text']}"
        for s in segments
    ]
    timestamped_text = "\n".join(timestamped_lines)
    return {
        "text": text,
        "segments": segments,
        "timestamped_text": timestamped_text,
        "backend": "faster-whisper",
        "model": WHISPER_MODEL,
        "language": info.language,
        "chunked": False,
    }
