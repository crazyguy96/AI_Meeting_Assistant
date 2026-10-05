import json
from typing import Any

from app.core.clients import get_groq_client
from app.core.config import GROQ_DOCUMENTATION_MODEL
from app.core.utils import render_markdown
from app.pipeline.prompts import load_prompt


def generate_record(refined: dict[str, Any]) -> dict[str, Any]:
    response = get_groq_client().chat.completions.create(
        model=GROQ_DOCUMENTATION_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": load_prompt("documentation_prompt.txt"),
            },
            {
                "role": "user",
                "content": "Create the final meeting record from:\n"
                + refined["refined_text"],
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Groq model {GROQ_DOCUMENTATION_MODEL} returned an empty response."
        )
    return json.loads(content)


__all__ = ["generate_record", "render_markdown"]
