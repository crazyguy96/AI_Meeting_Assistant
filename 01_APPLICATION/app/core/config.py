import os
from pathlib import Path

from dotenv import load_dotenv

# Repository directory paths for prompt templates and generated run outputs.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "02_PROMPTS"
OUTPUT_DIR = PROJECT_ROOT / "01_APPLICATION" / "outputs"

load_dotenv(PROJECT_ROOT / ".env")

def is_cuda_available() -> bool:
    """Checks whether CUDA acceleration is available via ctranslate2 or torch."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def get_whisper_config() -> tuple[str, str, str]:
    """Resolves Whisper model, device, and compute type from env vars or hardware defaults."""
    model = (os.getenv("WHISPER_MODEL") or "small").strip()

    device = os.getenv("WHISPER_DEVICE")
    if device:
        device = device.strip()
    else:
        device = "cuda" if is_cuda_available() else "cpu"

    compute_type = os.getenv("WHISPER_COMPUTE_TYPE")
    if compute_type:
        compute_type = compute_type.strip()
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return model, device, compute_type


# Local faster-whisper model settings for offline audio transcription
WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = get_whisper_config()

# Groq LLM model identifiers for transcript refinement and documentation generation.
GROQ_REFINEMENT_MODEL = os.getenv(
    "GROQ_REFINEMENT_MODEL", "openai/gpt-oss-120b"
)
GROQ_DOCUMENTATION_MODEL = os.getenv(
    "GROQ_DOCUMENTATION_MODEL", "openai/gpt-oss-20b"
)

# Audio validation thresholds and constraints
AUDIO_MIN_DURATION_SECONDS = float(os.getenv("AUDIO_MIN_DURATION_SECONDS", "0.5"))
AUDIO_MAX_DURATION_SECONDS = float(os.getenv("AUDIO_MAX_DURATION_SECONDS", "14400.0"))
AUDIO_SILENCE_THRESHOLD = float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.001"))
AUDIO_SUSPICIOUS_MIN_SIZE_BYTES = int(os.getenv("AUDIO_SUSPICIOUS_MIN_SIZE_BYTES", "1024"))
AUDIO_SUSPICIOUS_MAX_SIZE_BYTES = int(os.getenv("AUDIO_SUSPICIOUS_MAX_SIZE_BYTES", str(500 * 1024 * 1024)))


# Retrieves a required API key from environment variables or raises an informative error.
def require_api_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Add it to the repository-root .env file."
        )
    return value
