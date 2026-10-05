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
    return {
        "text": text,
        "segments": segments,
        "backend": "faster-whisper",
        "model": WHISPER_MODEL,
        "language": info.language,
        "chunked": False,
    }
