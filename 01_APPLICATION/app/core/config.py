import os
from pathlib import Path

from dotenv import load_dotenv

# Repository directory paths for prompt templates and generated run outputs.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "02_PROMPTS"
OUTPUT_DIR = PROJECT_ROOT / "01_APPLICATION" / "outputs"

load_dotenv(PROJECT_ROOT / ".env")

# Local faster wisper model settings for offline audio transcription
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Groq LLM model identifiers for transcript refinement and documentation generation.
GROQ_REFINEMENT_MODEL = os.getenv(
    "GROQ_REFINEMENT_MODEL", "openai/gpt-oss-120b"
)
GROQ_DOCUMENTATION_MODEL = os.getenv(
    "GROQ_DOCUMENTATION_MODEL", "openai/gpt-oss-20b"
)


# Retrieves a required API key from environment variables or raises an informative error.
def require_api_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Add it to the repository-root .env file."
        )
    return value
