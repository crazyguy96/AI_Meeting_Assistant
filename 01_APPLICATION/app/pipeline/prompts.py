from functools import lru_cache

from app.core.config import PROMPTS_DIR


# Reads, validates, and caches prompt text files from the 02_PROMPTS directory.
@lru_cache(maxsize=3)
def load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Required model prompt is missing: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Model prompt is empty: {prompt_path}")
    return prompt
