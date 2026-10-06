from functools import lru_cache
import inspect
import logging
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from app.core import config
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


# Loads and caches the local Faster-Whisper model using configured hardware acceleration and quantization.
@lru_cache(maxsize=1)
def get_whisper_model(
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    model_name = model or config.WHISPER_MODEL
    device_name = device or config.WHISPER_DEVICE
    compute_name = compute_type or config.WHISPER_COMPUTE_TYPE
    logger.info(
        "Loading faster-whisper model %s on %s with %s compute",
        model_name,
        device_name,
        compute_name,
    )
    return WhisperModel(
        model_name,
        device=device_name,
        compute_type=compute_name,
    )


# Formats a user glossary (string or collection of terms) into an initial_prompt for Whisper.
def format_glossary_prompt(glossary: Any) -> str | None:
    if not glossary:
        return None
    if isinstance(glossary, (list, tuple, set)):
        terms = [str(term).strip() for term in glossary if str(term).strip()]
    elif isinstance(glossary, str):
        terms = [term.strip() for term in glossary.split(",") if term.strip()]
    else:
        terms = [str(glossary).strip()] if str(glossary).strip() else []

    deduped = list(dict.fromkeys(terms))
    return ", ".join(deduped) if deduped else None


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
def transcribe_audio(
    file_path: str | Path,
    glossary: str | list[str] | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    logger.info("Received uploaded audio for transcription: %s", Path(file_path).name)
    path = validate_audio_file(file_path)

    model_name = config.WHISPER_MODEL
    device_name = config.WHISPER_DEVICE
    compute_name = config.WHISPER_COMPUTE_TYPE

    logger.info(
        "Loading/reusing faster-whisper model %s on %s (%s)",
        model_name,
        device_name,
        compute_name,
    )
    model = get_whisper_model()

    prompt = (
        initial_prompt.strip()
        if isinstance(initial_prompt, str) and initial_prompt.strip()
        else None
    )
    if not prompt and glossary is not None:
        prompt = format_glossary_prompt(glossary)

    logger.info(
        "Starting English transcription: %s (vad_filter=True, initial_prompt=%s)",
        path.name,
        repr(prompt) if prompt else "None",
    )

    transcribe_kwargs: dict[str, Any] = {
        "language": "en",
        "vad_filter": True,
    }
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt

    try:
        sig = inspect.signature(model.transcribe)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_kwargs:
            filtered_kwargs = {
                k: v for k, v in transcribe_kwargs.items() if k in sig.parameters
            }
        else:
            filtered_kwargs = transcribe_kwargs
        segments_iterator, info = model.transcribe(str(path), **filtered_kwargs)
    except TypeError:
        segments_iterator, info = model.transcribe(str(path), language="en")

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
        "model": model_name,
        "language": info.language,
        "chunked": False,
    }
