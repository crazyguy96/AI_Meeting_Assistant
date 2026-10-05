from functools import lru_cache

from app.core.config import require_api_key


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq

    return Groq(api_key=require_api_key("GROQ_API_KEY"))
