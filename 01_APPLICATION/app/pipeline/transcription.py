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


# Validates that the uploaded recording exists, has an allowed format, is non-empty, decodable, non-silent, and within duration bounds.
def validate_audio_file(file_path: str | Path) -> Path:
    path = Path(file_path)
    logger.info("Validating uploaded audio: %s", path.name)
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio file type. Supported extensions: {supported}.")
    if not path.is_file():
        raise ValueError("The uploaded recording is missing or is not a file.")
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError("The uploaded recording is empty.")
        with path.open("rb") as audio_file:
            audio_file.read(1)
    except OSError as exc:
        raise ValueError(f"The uploaded recording cannot be read: {path.name}") from exc

    # File size warning for suspiciously small/large files
    min_size = getattr(config, "AUDIO_SUSPICIOUS_MIN_SIZE_BYTES", 1024)
    max_size = getattr(config, "AUDIO_SUSPICIOUS_MAX_SIZE_BYTES", 500 * 1024 * 1024)
    if file_size < min_size:
        logger.warning(
            "The uploaded audio file %s is suspiciously small (%d bytes).",
            path.name,
            file_size,
        )
    elif file_size > max_size:
        logger.warning(
            "The uploaded audio file %s is suspiciously large (%d bytes).",
            path.name,
            file_size,
        )

    # Validate audio can be opened and probed by audio decoder (PyAV)
    try:
        import av
        with av.open(str(path)) as container:
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                raise ValueError(
                    f"The uploaded recording contains no audio streams: {path.name}"
                )
            stream = audio_streams[0]

            duration = None
            if container.duration is not None and container.duration > 0:
                duration = float(container.duration) / av.time_base
            elif stream.duration is not None and stream.time_base is not None and stream.duration > 0:
                duration = float(stream.duration * stream.time_base)

            min_duration = getattr(config, "AUDIO_MIN_DURATION_SECONDS", 0.5)
            max_duration = getattr(config, "AUDIO_MAX_DURATION_SECONDS", 14400.0)

            if duration is not None:
                if duration < min_duration:
                    raise ValueError(
                        f"The uploaded recording duration ({duration:.2f}s) is below the minimum allowed duration ({min_duration}s)."
                    )
                if duration > max_duration:
                    raise ValueError(
                        f"The uploaded recording duration ({duration:.2f}s) exceeds the maximum allowed duration ({max_duration}s)."
                    )

            # Silence and frame decodability check
            silence_threshold = getattr(config, "AUDIO_SILENCE_THRESHOLD", 0.001)
            max_amp = 0.0
            frames_checked = 0
            has_sound = False
            total_duration_decoded = 0.0

            for frame in container.decode(stream):
                frames_checked += 1
                arr = frame.to_ndarray()
                if arr.size > 0:
                    import numpy as np
                    if np.issubdtype(arr.dtype, np.floating):
                        frame_amp = float(np.abs(arr).max())
                    elif arr.dtype == np.int16:
                        frame_amp = float(np.abs(arr).max()) / 32768.0
                    elif arr.dtype == np.int32:
                        frame_amp = float(np.abs(arr).max()) / 2147483648.0
                    else:
                        frame_amp = float(np.abs(arr).max()) / 32768.0

                    if frame_amp > max_amp:
                        max_amp = frame_amp
                    if max_amp >= silence_threshold:
                        has_sound = True
                        break

                if frame.time is not None:
                    total_duration_decoded = float(frame.time)
                # Allow leading digital silence by continuing to scan frames until sound is detected.
                # Bound pure-silence scan to 50,000 frames (~15 minutes) to protect against infinite silent streams.
                if frames_checked >= 50000:
                    break

            if frames_checked == 0:
                raise ValueError(
                    f"The uploaded audio file contains no decodable audio frames: {path.name}"
                )

            if not has_sound and max_amp < silence_threshold:
                raise ValueError(
                    f"The uploaded recording is completely silent or near-silent: {path.name}"
                )

            if duration is None:
                duration = total_duration_decoded
                if duration < min_duration:
                    raise ValueError(
                        f"The uploaded recording duration ({duration:.2f}s) is below the minimum allowed duration ({min_duration}s)."
                    )
    except (av.error.FFmpegError, av.error.InvalidDataError) as exc:
        raise ValueError(
            f"The uploaded audio file is corrupt or cannot be decoded: {path.name}"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"The uploaded audio file is corrupt or cannot be decoded: {path.name}"
        ) from exc

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
