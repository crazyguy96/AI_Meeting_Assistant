import logging
from typing import Any

from app.core.config import GROQ_DOCUMENTATION_MODEL
from app.core.utils import render_markdown, sanitize_exception
from app.pipeline.evidence import normalize_match
from app.pipeline.prompts import load_prompt
from app.pipeline.refinement import (
    _split_text,
    _token_count,
    groq_json,
)

logger = logging.getLogger(__name__)


# Deterministically merges structured records from multiple chunks, deduplicating items without inventing fields.
def merge_records(chunk_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunk_records:
        return {
            "meeting_title": "Meeting",
            "summary": "",
            "minutes": [],
            "decisions": [],
            "non_decisions": [],
            "action_items": [],
            "discussion_points": [],
            "open_questions": [],
        }
    if len(chunk_records) == 1:
        return chunk_records[0]

    meeting_title = "Meeting"
    for r in chunk_records:
        t = str(r.get("meeting_title") or "").strip()
        if t and t.lower() not in ("meeting", "short title"):
            meeting_title = t
            break
    if meeting_title == "Meeting" and chunk_records[0].get("meeting_title"):
        meeting_title = str(chunk_records[0]["meeting_title"]).strip()

    summaries: list[str] = []
    seen_summaries: set[str] = set()
    for r in chunk_records:
        s = str(r.get("summary") or "").strip()
        if s and s not in seen_summaries:
            seen_summaries.add(s)
            summaries.append(s)
    merged_summary = " ".join(summaries)

    merged_minutes: list[str] = []
    seen_minutes: set[str] = set()
    for r in chunk_records:
        for m in r.get("minutes") or []:
            text_m = str(m).strip()
            norm_m = normalize_match(text_m)
            if norm_m and norm_m not in seen_minutes:
                seen_minutes.add(norm_m)
                merged_minutes.append(text_m)

    merged_decisions: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    for r in chunk_records:
        for d in r.get("decisions") or []:
            if not isinstance(d, dict):
                continue
            text_d = str(d.get("text") or "").strip()
            norm_key = normalize_match(text_d) or normalize_match(d.get("evidence_quote", ""))
            if norm_key and norm_key not in seen_decisions:
                seen_decisions.add(norm_key)
                merged_decisions.append(d)

    merged_non_decisions: list[dict[str, Any]] = []
    seen_non_decisions: set[str] = set()
    for r in chunk_records:
        for nd in r.get("non_decisions") or []:
            if not isinstance(nd, dict):
                continue
            text_nd = str(nd.get("text") or "").strip()
            norm_key = normalize_match(text_nd) or normalize_match(nd.get("evidence_quote", ""))
            if norm_key and norm_key not in seen_non_decisions and norm_key not in seen_decisions:
                seen_non_decisions.add(norm_key)
                merged_non_decisions.append(nd)

    merged_actions: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for r in chunk_records:
        for a in r.get("action_items") or []:
            if not isinstance(a, dict):
                continue
            task_str = str(a.get("task") or "").strip()
            norm_key = normalize_match(task_str) or normalize_match(a.get("evidence_quote", ""))
            if norm_key and norm_key not in seen_actions:
                seen_actions.add(norm_key)
                cleaned_action = dict(a)
                cleaned_action["owner"] = str(cleaned_action.get("owner") or "Unspecified")
                cleaned_action["deadline"] = str(cleaned_action.get("deadline") or "Unspecified")
                merged_actions.append(cleaned_action)

    merged_discussions: list[str] = []
    seen_discussions: set[str] = set()
    for r in chunk_records:
        for dp in r.get("discussion_points") or []:
            text_dp = str(dp).strip()
            norm_dp = normalize_match(text_dp)
            if norm_dp and norm_dp not in seen_discussions:
                seen_discussions.add(norm_dp)
                merged_discussions.append(text_dp)

    merged_questions: list[str] = []
    seen_questions: set[str] = set()
    for r in chunk_records:
        for oq in r.get("open_questions") or []:
            text_oq = str(oq).strip()
            norm_oq = normalize_match(text_oq)
            if norm_oq and norm_oq not in seen_questions:
                seen_questions.add(norm_oq)
                merged_questions.append(text_oq)

    return {
        "meeting_title": meeting_title,
        "summary": merged_summary,
        "minutes": merged_minutes,
        "decisions": merged_decisions,
        "non_decisions": merged_non_decisions,
        "action_items": merged_actions,
        "discussion_points": merged_discussions,
        "open_questions": merged_questions,
    }


# Converts the refined transcript into a structured meeting record using chunking and Groq GPT-OSS-20B.
def generate_record(refined: dict[str, Any]) -> dict[str, Any]:
    print("[STAGE] documentation", flush=True)
    refined_text = str(refined.get("refined_text") or "").strip()
    if not refined_text:
        return {
            "meeting_title": "Meeting",
            "summary": "No speech detected.",
            "minutes": [],
            "decisions": [],
            "non_decisions": [],
            "action_items": [],
            "discussion_points": [],
            "open_questions": [],
        }

    system_prompt = load_prompt("documentation_prompt.txt")
    make_prompt = lambda chunk: "Create the final meeting record from:\n" + chunk

    chunks = _split_text(refined_text, system_prompt, make_prompt)
    total_chunks = len(chunks)

    chunk_records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, 1):
        user_prompt = make_prompt(chunk)
        input_tokens = _token_count(system_prompt) + _token_count(user_prompt)
        print(f"[DOC] chunk={idx}/{total_chunks} input_tokens={input_tokens}", flush=True)
        logger.info("[DOC] chunk=%d/%d input_tokens=%d", idx, total_chunks, input_tokens)

        try:
            record = groq_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=GROQ_DOCUMENTATION_MODEL,
                max_tokens=2048,
                retries=5,
            )
            chunk_records.append(record)
        except Exception as exc:
            chunk_info = f"chunk={idx}/{total_chunks} " if total_chunks > 1 else ""
            print(f"[ERROR] stage=documentation {chunk_info}model={GROQ_DOCUMENTATION_MODEL}", flush=True)
            print(f"[ERROR] exception={sanitize_exception(exc)}", flush=True)
            raise

    return merge_records(chunk_records)


__all__ = ["generate_record", "merge_records", "render_markdown"]
