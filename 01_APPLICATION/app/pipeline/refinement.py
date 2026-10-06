import inspect
import json
import logging
import re
import time
from functools import lru_cache
from typing import Any, Callable

import tiktoken

from app.core.clients import get_groq_client
from app.core.config import GROQ_REFINEMENT_MODEL
from app.core.utils import sanitize_exception
from app.pipeline.evidence import safe_edit
from app.pipeline.prompts import load_prompt

logger = logging.getLogger(__name__)

# Token limits and safety margins for Groq LLM requests.
MAX_GROQ_INPUT_TOKENS = 2600
CHAT_TOKEN_OVERHEAD = 128
MAX_GROQ_OUTPUT_TOKENS = 2048

# Strict JSON schema for Groq structured glossary extraction.
GLOSSARY_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_glossary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of domain terms, names, and technical terms extracted from the transcript",
                }
            },
            "required": ["terms"],
            "additionalProperties": False,
        },
    },
}


# Caches and returns the tiktoken tokenizer for OpenAI/o200k-compatible token counting.
@lru_cache(maxsize=1)
def _token_encoding() -> Any:
    return tiktoken.get_encoding("o200k_base")


# Computes the exact token count using tiktoken with graceful fallback when tokenizer data is unavailable.
def _token_count(text: str) -> int:
    try:
        return len(_token_encoding().encode(text))
    except Exception as exc:
        logger.warning(
            "tiktoken tokenizer unavailable (%s); falling back to heuristic token count.",
            sanitize_exception(exc),
        )
        return max(1, len(text) // 4)


# Checks whether combined system and user prompts fit within Groq's maximum input token budget.
def _fits_prompt(system_prompt: str, user_prompt: str) -> bool:
    return (
        _token_count(system_prompt)
        + _token_count(user_prompt)
        + CHAT_TOKEN_OVERHEAD
        <= MAX_GROQ_INPUT_TOKENS
    )


# Splits a long text on word boundaries into token-safe chunks before sending them to Groq.
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


# Groups transcript segments into token-safe batches before requesting Groq refinement.
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


# Formats transcript segments into a multiline string with bracketed start-to-end timestamps.
def format_timestamped(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return ""
    return "\n".join(
        f"[{format_time(s.get('start', 0))} – {format_time(s.get('end', 0))}] {s.get('text', '').strip()}"
        for s in segments
        if s.get("text", "").strip()
    )


# Executes a structured JSON chat completion with Groq, managing retries, backoff, and JSON schema fallbacks.
def groq_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = MAX_GROQ_OUTPUT_TOKENS,
    retries: int = 3,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_err: Exception | None = None

    # Groq JSON mode requires the prompt to explicitly request JSON.
    json_system_prompt = (
        system_prompt.strip()
        + "\n\n"
        "IMPORTANT: Return ONLY a valid JSON object. "
        "Do not return Markdown, code fences, explanations, or any text outside the JSON object."
    )

    req_format = response_format if response_format is not None else {"type": "json_object"}

    for attempt in range(retries):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": req_format,
            "messages": [
                {
                    "role": "system",
                    "content": json_system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        # Restrict runaway reasoning tokens for reasoning models (e.g. gpt-oss)
        if "gpt-oss" in model.lower() or "deepseek-r1" in model.lower():
            create_kwargs["reasoning_effort"] = "low"

        try:
            try:
                response = get_groq_client().chat.completions.create(**create_kwargs)
            except TypeError as te:
                if "reasoning_effort" in str(te) and "reasoning_effort" in create_kwargs:
                    create_kwargs.pop("reasoning_effort", None)
                    response = get_groq_client().chat.completions.create(**create_kwargs)
                else:
                    raise

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    f"Groq model {model} returned an empty response."
                )

            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    return json.loads(cleaned[start_idx : end_idx + 1])
                raise

        except Exception as exc:
            last_err = exc
            exc_str = str(exc).lower()

            # If json_schema failed with a validation error, fall back to json_object
            if req_format.get("type") == "json_schema" and (
                "json_validate_failed" in exc_str
                or "400" in exc_str
                or "schema" in exc_str
            ):
                logger.warning(
                    "Groq json_schema validation failed on attempt %d: %s. Falling back to json_object.",
                    attempt + 1,
                    sanitize_exception(exc),
                )
                req_format = {"type": "json_object"}
                continue

            # If json_validate_failed happens on json_object, retry with backoff
            if "json_validate_failed" in exc_str and attempt < retries - 1:
                wait = (attempt + 1) * 2.0
                logger.warning(
                    "Groq JSON validation failed on attempt %d: %s. Retrying in %.1fs...",
                    attempt + 1,
                    sanitize_exception(exc),
                    wait,
                )
                time.sleep(wait)
                continue

            # Do NOT retry oversized 413 requests
            if any(term in exc_str for term in ("413", "payload too large", "request too large", "entity too large")):
                raise

            # Retry only transient errors: 429, 5xx, rate limits, TPM, network timeouts
            is_transient = any(
                term in exc_str
                for term in (
                    "rate limit",
                    "tpm",
                    "429",
                    "too many requests",
                    "connection",
                    "timeout",
                    "500",
                    "502",
                    "503",
                    "504",
                    "internal server error",
                    "bad gateway",
                    "service unavailable",
                    "gateway timeout",
                )
            )

            if is_transient and attempt < retries - 1:
                match = re.search(r"(?:try again in|retry after)\s+([\d\.]+)\s*(ms|s)?", str(exc), re.IGNORECASE)
                if match:
                    try:
                        raw_val = float(match.group(1))
                        unit = (match.group(2) or "s").lower()
                        if unit == "ms":
                            raw_val = raw_val / 1000.0
                        wait = max(1.0, raw_val + 1.0)
                    except (ValueError, TypeError):
                        wait = (attempt + 1) * 4.0
                else:
                    wait = (attempt + 1) * 4.0

                logger.warning(
                    "Groq transient error on attempt %d: %s. Retrying in %.1fs...",
                    attempt + 1,
                    sanitize_exception(exc),
                    wait,
                )

                time.sleep(wait)
            else:
                raise

    if last_err:
        raise last_err

    raise RuntimeError("groq_json failed without explicit exception.")


# Extracts domain terminology, names, and technical acronyms from the transcript using Groq.
def extract_glossary(
    transcript: str, user_glossary: list[str] | None = None
) -> list[str]:
    print("[STAGE] glossary", flush=True)
    user_terms = list(dict.fromkeys(user_glossary or []))
    if not transcript or not transcript.strip():
        return user_terms

    system_prompt = load_prompt("glossary_prompt.txt")
    make_prompt = lambda text: "Extract terminology:\n" + text

    try:
        chunks = _split_text(transcript, system_prompt, make_prompt)
    except Exception as exc:
        print(f"[ERROR] stage=glossary model={GROQ_REFINEMENT_MODEL}", flush=True)
        print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
        logger.warning(
            "Glossary chunking failed (%s). Continuing refinement without optional glossary enrichment.",
            sanitize_exception(exc),
        )
        return user_terms

    kwargs: dict[str, Any] = {"max_tokens": 1024}
    try:
        if "response_format" in inspect.signature(groq_json).parameters:
            kwargs["response_format"] = GLOSSARY_SCHEMA
    except (ValueError, TypeError):
        pass

    terms: list[str] = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue
        try:
            result = groq_json(
                system_prompt,
                make_prompt(chunk),
                GROQ_REFINEMENT_MODEL,
                **kwargs,
            )
            raw_terms = result.get("terms", []) if isinstance(result, dict) else []
            if isinstance(raw_terms, list):
                terms.extend(
                    str(t).strip()
                    for t in raw_terms
                    if t is not None and str(t).strip()
                )
            else:
                logger.warning(
                    "Glossary extraction returned non-list 'terms': %r. Continuing without these terms.",
                    raw_terms,
                )
        except Exception as exc:
            print(f"[ERROR] stage=glossary chunk={idx}/{total_chunks} model={GROQ_REFINEMENT_MODEL}", flush=True)
            print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
            logger.warning(
                "Glossary extraction failed (%s). Continuing refinement without optional glossary enrichment.",
                sanitize_exception(exc),
            )

    return list(dict.fromkeys(user_terms + terms))


# Refines the raw transcript with Groq GPT-OSS-120B while applying only verified, safe edits.
def refine_transcription(
    transcription: dict[str, Any], user_glossary: list[str] | None = None
) -> dict[str, Any]:
    raw_text = transcription.get("text", "")
    raw_segments = transcription.get("segments") or []
    if not raw_segments and raw_text:
        raw_segments = [{"id": "S0001", "start": 0.0, "end": 0.0, "text": raw_text}]

    try:
        glossary = extract_glossary(raw_text, user_glossary)
    except Exception as exc:
        print(f"[ERROR] stage=glossary model={GROQ_REFINEMENT_MODEL}", flush=True)
        print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
        logger.warning(
            "Glossary extraction raised unexpected error (%s). Falling back to user glossary.",
            sanitize_exception(exc),
        )
        glossary = list(dict.fromkeys(user_glossary or []))

    system_prompt = load_prompt("refinement_prompt.txt")

    def make_prompt(segments: list[dict[str, Any]]) -> str:
        return (
            "GLOSSARY:\n"
            + json.dumps(glossary, ensure_ascii=False)
            + "\nSEGMENTS:\n"
            + json.dumps(segments, ensure_ascii=False)
        )

    batches = _chunk_segments(
        raw_segments,
        system_prompt,
        make_prompt,
    )

    segments = [dict(segment) for segment in raw_segments]
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    proposed_edits: list[dict[str, Any]] = []
    total_batches = len(batches)
    for batch_idx, batch in enumerate(batches, 1):
        print(f"[STAGE] refinement chunk {batch_idx}/{total_batches}", flush=True)
        try:
            result = groq_json(
                system_prompt,
                make_prompt(batch),
                GROQ_REFINEMENT_MODEL,
            )
        except Exception as exc:
            print(f"[ERROR] stage=refinement chunk={batch_idx}/{total_batches} model={GROQ_REFINEMENT_MODEL}", flush=True)
            print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
            raise
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
    refined_timestamped = format_timestamped(segments) or refined_text
    return {
        "refined_text": refined_text,
        "refined_segments": segments,
        "refined_timestamped": refined_timestamped,
        "glossary": glossary,
        "proposed_edits": proposed_edits,
        "applied_edits": applied,
        "rejected_edits": rejected,
        "backend": "Groq",
        "model": GROQ_REFINEMENT_MODEL,
    }
