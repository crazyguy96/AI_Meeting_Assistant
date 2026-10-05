import json
from functools import lru_cache
from typing import Any, Callable

import tiktoken

from app.core.clients import get_groq_client
from app.core.config import GROQ_REFINEMENT_MODEL
from app.pipeline.evidence import safe_edit
from app.pipeline.prompts import load_prompt

MAX_GROQ_INPUT_TOKENS = 5200
CHAT_TOKEN_OVERHEAD = 128
MAX_GROQ_OUTPUT_TOKENS = 1800


@lru_cache(maxsize=1)
def _token_encoding() -> Any:
    return tiktoken.get_encoding("o200k_base")


def _token_count(text: str) -> int:
    return len(_token_encoding().encode(text))


def _fits_prompt(system_prompt: str, user_prompt: str) -> bool:
    return (
        _token_count(system_prompt)
        + _token_count(user_prompt)
        + CHAT_TOKEN_OVERHEAD
        <= MAX_GROQ_INPUT_TOKENS
    )


def _split_text(
    text: str,
    system_prompt: str,
    make_prompt: Callable[[str], str],
) -> list[str]:
    if _fits_prompt(system_prompt, make_prompt(text)):
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        best = start
        while low <= high:
            end = (low + high) // 2
            if _fits_prompt(system_prompt, make_prompt(text[start:end])):
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == start:
            raise ValueError(
                "A transcript character cannot fit within the configured Groq token budget."
            )
        if best < len(text):
            boundary = text.rfind(" ", start, best)
            if boundary > start:
                best = boundary + 1
        chunks.append(text[start:best])
        start = best
    if "".join(chunks) != text:
        raise RuntimeError("Transcript chunking did not preserve the source text.")
    return chunks


def _chunk_segments(
    segments: list[dict[str, Any]],
    system_prompt: str,
    make_prompt: Callable[[list[dict[str, Any]]], str],
) -> list[list[dict[str, Any]]]:
    parts: list[dict[str, Any]] = []
    for segment in segments:
        segment = dict(segment)
        if _fits_prompt(system_prompt, make_prompt([segment])):
            parts.append(segment)
            continue
        text_parts = _split_text(
            str(segment["text"]),
            system_prompt,
            lambda text: make_prompt([{**segment, "text": text}]),
        )
        parts.extend({**segment, "text": text} for text in text_parts)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in parts:
        candidate = current + [segment]
        if _fits_prompt(system_prompt, make_prompt(candidate)):
            current = candidate
        else:
            if current:
                batches.append(current)
            if not _fits_prompt(system_prompt, make_prompt([segment])):
                raise RuntimeError("Transcript segment exceeds the Groq token budget.")
            current = [segment]
    if current:
        batches.append(current)
    return batches


def groq_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = MAX_GROQ_OUTPUT_TOKENS,
) -> dict[str, Any]:
    response = get_groq_client().chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Groq model {model} returned an empty response.")
    return json.loads(content)


def extract_glossary(
    transcript: str, user_glossary: list[str] | None = None
) -> list[str]:
    system_prompt = load_prompt("glossary_prompt.txt")
    make_prompt = lambda text: "Extract terminology:\n" + text
    chunks = _split_text(transcript, system_prompt, make_prompt)
    terms: list[str] = []
    for chunk in chunks:
        result = groq_json(
            system_prompt,
            make_prompt(chunk),
            GROQ_REFINEMENT_MODEL,
            max_tokens=512,
        )
        terms.extend(result.get("terms", []))
    return list(dict.fromkeys((user_glossary or []) + terms))


def refine_transcription(
    transcription: dict[str, Any], user_glossary: list[str] | None = None
) -> dict[str, Any]:
    glossary = extract_glossary(transcription["text"], user_glossary)
    system_prompt = load_prompt("refinement_prompt.txt")

    def make_prompt(segments: list[dict[str, Any]]) -> str:
        return (
            "GLOSSARY:\n"
            + json.dumps(glossary, ensure_ascii=False)
            + "\nSEGMENTS:\n"
            + json.dumps(segments, ensure_ascii=False)
        )

    batches = _chunk_segments(
        transcription["segments"],
        system_prompt,
        make_prompt,
    )

    segments = [dict(segment) for segment in transcription["segments"]]
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    proposed_edits: list[dict[str, Any]] = []
    for batch in batches:
        result = groq_json(
            system_prompt,
            make_prompt(batch),
            GROQ_REFINEMENT_MODEL,
        )
        for edit in result.get("edits", []):
            proposed_edits.append(edit)
            source = str(edit.get("from", ""))
            target = str(edit.get("to", ""))
            confidence = float(edit.get("confidence", 0))
            safe, reasons = safe_edit(source, target)
            if not safe or confidence < 0.75:
                rejected.append(
                    {**edit, "reasons": reasons or ["low confidence"]}
                )
                continue
            for segment in segments:
                if segment["id"] == edit.get("segment_id") and source in segment["text"]:
                    segment["text"] = segment["text"].replace(source, target, 1)
                    applied.append(edit)
                    break
            else:
                rejected.append({**edit, "reasons": ["exact source text not found"]})

    refined_text = " ".join(segment["text"] for segment in segments)
    return {
        "refined_text": refined_text,
        "refined_segments": segments,
        "refined_timestamped": refined_text,
        "glossary": glossary,
        "proposed_edits": proposed_edits,
        "applied_edits": applied,
        "rejected_edits": rejected,
        "backend": "Groq",
        "model": GROQ_REFINEMENT_MODEL,
    }
