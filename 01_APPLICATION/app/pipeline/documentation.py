import difflib
import logging
from typing import Any

from app.core.config import (
    DOCUMENTATION_CHUNK_OVERLAP_WORDS,
    DOCUMENTATION_MAX_INPUT_TOKENS,
    DOCUMENTATION_MAX_OUTPUT_TOKENS,
    GROQ_DOCUMENTATION_MODEL,
)
from app.core.utils import render_markdown, sanitize_exception
from app.pipeline.evidence import (
    _extract_name_tokens,
    normalize_match,
    number_words,
    protected_tokens,
)
from app.pipeline.prompts import load_prompt
from app.pipeline.refinement import (
    _split_text,
    _token_count,
    groq_json,
)

logger = logging.getLogger(__name__)


# Determines whether two texts are near-duplicates using token similarity and SequenceMatcher.
# Guarded by protected_tokens, number_words, and entity-name checks so items with different dates,
# numbers, negations, or entities are never merged.
def are_near_duplicate_texts(t1: str, t2: str, threshold: float = 0.82) -> bool:
    norm1 = normalize_match(t1)
    norm2 = normalize_match(t2)
    if not norm1 or not norm2:
        return False
    if norm1 == norm2:
        return True
    if protected_tokens(t1) != protected_tokens(t2):
        return False
    if number_words(t1) != number_words(t2):
        return False
    names1 = sorted(_extract_name_tokens(t1))
    names2 = sorted(_extract_name_tokens(t2))
    if names1 != names2:
        return False
    ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    if ratio >= threshold:
        return True
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 and words2:
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        jaccard = len(words1 & words2) / len(words1 | words2)
        if overlap >= 0.85 and jaccard >= 0.70:
            return True
    return False


# Deterministically merges structured records from multiple chunks, deduplicating near-identical items without inventing fields.
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
            # If the model emitted a run-on semicolon string, split it into separate topic bullets
            if ";" in text_m and len(text_m.split(";")) >= 3:
                parts = [p.strip() for p in text_m.split(";") if p.strip()]
            else:
                parts = [text_m]
            for part in parts:
                norm_m = normalize_match(part)
                if norm_m and norm_m not in seen_minutes:
                    seen_minutes.add(norm_m)
                    merged_minutes.append(part)

    merged_decisions: list[dict[str, Any]] = []
    for r in chunk_records:
        for d in r.get("decisions") or []:
            if not isinstance(d, dict):
                continue
            text_d = str(d.get("text") or "").strip()
            if not text_d:
                continue
            matched_idx = None
            for idx, existing in enumerate(merged_decisions):
                if are_near_duplicate_texts(existing.get("text", ""), text_d):
                    matched_idx = idx
                    break
            if matched_idx is not None:
                ex = merged_decisions[matched_idx]
                longer_text = text_d if len(text_d) > len(ex.get("text", "")) else ex.get("text", "")
                ex_quote = ex.get("evidence_quote", "")
                d_quote = d.get("evidence_quote", "")
                longer_quote = d_quote if len(d_quote) > len(ex_quote) else ex_quote
                status = "Confirmed" if (ex.get("status") == "Confirmed" or d.get("status") == "Confirmed") else "Needs Review"
                merged_decisions[matched_idx] = {
                    "text": longer_text,
                    "status": status,
                    "evidence_quote": longer_quote,
                }
            else:
                merged_decisions.append(dict(d))

    merged_non_decisions: list[dict[str, Any]] = []
    for r in chunk_records:
        for nd in r.get("non_decisions") or []:
            if not isinstance(nd, dict):
                continue
            text_nd = str(nd.get("text") or "").strip()
            if not text_nd:
                continue
            matched_idx = None
            for idx, existing in enumerate(merged_non_decisions):
                if are_near_duplicate_texts(existing.get("text", ""), text_nd):
                    matched_idx = idx
                    break
            if matched_idx is None:
                merged_non_decisions.append(dict(nd))

    merged_actions: list[dict[str, Any]] = []
    for r in chunk_records:
        for a in r.get("action_items") or []:
            if not isinstance(a, dict):
                continue
            task_a = str(a.get("task") or "").strip()
            if not task_a:
                continue
            matched_idx = None
            for idx, existing in enumerate(merged_actions):
                if are_near_duplicate_texts(existing.get("task", ""), task_a):
                    ex_owner = str(existing.get("owner") or "Unspecified").lower()
                    a_owner = str(a.get("owner") or "Unspecified").lower()
                    if ex_owner not in ("unspecified", "not specified") and a_owner not in ("unspecified", "not specified"):
                        if ex_owner != a_owner:
                            continue
                    ex_deadline = str(existing.get("deadline") or "Unspecified").lower()
                    a_deadline = str(a.get("deadline") or "Unspecified").lower()
                    if ex_deadline not in ("unspecified", "not specified") and a_deadline not in ("unspecified", "not specified"):
                        if ex_deadline != a_deadline:
                            continue
                    matched_idx = idx
                    break
            if matched_idx is not None:
                ex = merged_actions[matched_idx]
                longer_task = task_a if len(task_a) > len(ex.get("task", "")) else ex.get("task", "")
                owner = (
                    a.get("owner")
                    if str(a.get("owner", "")).lower() not in ("unspecified", "not specified", "")
                    else ex.get("owner", "Unspecified")
                )
                deadline = (
                    a.get("deadline")
                    if str(a.get("deadline", "")).lower() not in ("unspecified", "not specified", "")
                    else ex.get("deadline", "Unspecified")
                )
                ex_quote = ex.get("evidence_quote", "")
                a_quote = a.get("evidence_quote", "")
                longer_quote = a_quote if len(a_quote) > len(ex_quote) else ex_quote
                status = "Confirmed" if (ex.get("status") == "Confirmed" or a.get("status") == "Confirmed") else "Needs Review"
                merged = {
                    "task": longer_task,
                    "owner": str(owner or "Unspecified"),
                    "deadline": str(deadline or "Unspecified"),
                    "status": status,
                    "evidence_quote": longer_quote,
                }
                if ex.get("owner_note") or a.get("owner_note"):
                    merged["owner_note"] = ex.get("owner_note") or a.get("owner_note")
                merged_actions[matched_idx] = merged
            else:
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

    def doc_fits_prompt(sys_p: str, user_p: str) -> bool:
        from app.pipeline.refinement import CHAT_TOKEN_OVERHEAD, _token_count
        return (
            _token_count(sys_p) + _token_count(user_p) + CHAT_TOKEN_OVERHEAD
            <= DOCUMENTATION_MAX_INPUT_TOKENS
        )

    chunks = _split_text(
        refined_text,
        system_prompt,
        make_prompt,
        fits_fn=doc_fits_prompt,
    )
    total_chunks = len(chunks)

    chunk_records: list[dict[str, Any]] = []
    overlap_budget = DOCUMENTATION_CHUNK_OVERLAP_WORDS

    for idx, chunk in enumerate(chunks, 1):
        if idx > 1 and overlap_budget > 0:
            prev_chunk = chunks[idx - 2]
            prev_words = prev_chunk.split()
            tail_words = prev_words[-overlap_budget:] if len(prev_words) > overlap_budget else prev_words
            overlap_context = " ".join(tail_words)
            user_prompt = (
                f"Preceding dialogue context (for reference only):\n{overlap_context}\n\n"
                f"Current meeting transcript segment to document:\n{chunk}"
            )
        else:
            user_prompt = make_prompt(chunk)

        input_tokens = _token_count(system_prompt) + _token_count(user_prompt)
        print(f"[DOC] chunk={idx}/{total_chunks} input_tokens={input_tokens}", flush=True)
        logger.info("[DOC] chunk=%d/%d input_tokens=%d", idx, total_chunks, input_tokens)

        try:
            record = groq_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=GROQ_DOCUMENTATION_MODEL,
                max_tokens=DOCUMENTATION_MAX_OUTPUT_TOKENS,
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
