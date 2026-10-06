import difflib
import re
from typing import Any

# Regex matching critical factual entities (currencies, numbers, percentages, dates, negations, commitments).
_PROTECTED_TOKEN_PATTERN = re.compile(
    r"(?:"
    r"(?:US\$|CA\$|AU\$|₹|Rs\.?|INR|USD|EUR|GBP|[$€£])\s*"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:lakh|lakhs|crore|crores|cr|lpa|million|billion))?%?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|cr|lpa|million|billion)\b"
    r"|\d[\d,]*(?:\.\d+)?%"
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"
    r"|\b\d[\d,]*(?:\.\d+)?\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|not|never|no|cannot|can't|won't|don't|didn't|"
    r"haven't|hasn't|isn't|aren't|wasn't|weren't|will|might|may|could|should|"
    r"must|shall|would|can|commit|committed|promise|promised|agree|agreed|"
    r"plan|planned|intend|intends|intended)\b"
    r")",
    re.IGNORECASE,
)


# Extracts and normalizes protected factual tokens (numbers, currencies, dates, negations) from text.
def protected_tokens(text: str) -> list[str]:
    return sorted(
        match.group(0).lower().replace(" ", "")
        for match in _PROTECTED_TOKEN_PATTERN.finditer(str(text))
    )


# Ensures proposed transcript edits do not modify, remove, or corrupt protected factual information.
def safe_edit(source: str, target: str) -> tuple[bool, list[str]]:
    unchanged = protected_tokens(source) == protected_tokens(target)
    return unchanged, [] if unchanged else ["protected information changed"]


# Normalizes a text string for fuzzy or substring matching by stripping punctuation and whitespace.
def normalize_match(text: str) -> str:
    text = re.sub(r"[^\w₹$%.]+", " ", str(text).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# Locates supporting transcript segments and timestamps for an evidence quote using fuzzy matching.
def find_evidence(
    quote: str, segments: list[dict[str, Any]], threshold: float = 0.80
) -> dict[str, Any]:
    normalized_quote = normalize_match(quote or "")
    if not normalized_quote:
        return {"found": False, "timestamp": "Unknown", "segment_ids": [], "score": 0}

    best: dict[str, Any] = {
        "found": False,
        "timestamp": "Unknown",
        "segment_ids": [],
        "score": 0,
    }
    for start in range(len(segments)):
        combined = ""
        segment_ids = []
        for end in range(start, min(start + 3, len(segments))):
            segment = segments[end]
            combined = (combined + " " + normalize_match(segment["text"])).strip()
            segment_ids.append(segment["id"])
            if normalized_quote in combined:
                return {
                    "found": True,
                    "timestamp": segments[start].get("start", "Unknown"),
                    "segment_ids": segment_ids,
                    "score": 1.0,
                }
            score = difflib.SequenceMatcher(None, normalized_quote, combined).ratio()
            if score > best["score"]:
                best = {
                    "found": score >= threshold,
                    "timestamp": segments[start].get("start", "Unknown"),
                    "segment_ids": segment_ids.copy(),
                    "score": round(score, 3),
                }
    if not best["found"]:
        best["timestamp"], best["segment_ids"] = "Unknown", []
    return best


# Regex matching tentative or non-committal words indicating a proposal rather than a confirmed decision.
_PROPOSAL_INDICATOR_PATTERN = re.compile(
    r"\b(?:suggest|suggests|suggested|suggestion|propose|proposes|proposed|proposal|"
    r"idea|option|consider|considering|might|could|maybe|"
    r"not made a final decision|have not made a final decision|not yet decided|"
    r"haven't decided|has not decided|undecided|under discussion|review in next)\b",
    re.IGNORECASE,
)


# Allowed status values defined by the meeting documentation schema.
ALLOWED_DECISION_STATUSES = {"Confirmed", "Needs Review"}
ALLOWED_ACTION_STATUSES = {"Confirmed", "Needs Review"}


# Validates action item assignments, resetting ungrounded owners or deadlines to 'Unspecified'.
# Supports checking surrounding transcript context if an owner or deadline was confirmed across adjacent lines.
def validate_action(action: dict[str, Any], context_text: str = "") -> dict[str, Any]:
    validated = dict(action)
    validated["owner"] = str(validated.get("owner") or "Unspecified")
    validated["deadline"] = str(validated.get("deadline") or "Unspecified")

    # Validate action status against allowed values; normalize unsupported to "Needs Review"
    raw_status = str(validated.get("status") or "")
    validated["status"] = raw_status if raw_status in ALLOWED_ACTION_STATUSES else "Needs Review"

    evidence_quote = normalize_match(validated.get("evidence_quote", ""))
    normalized_context = normalize_match(context_text or "")
    for field in ("owner", "deadline"):
        value = validated[field]
        if value.lower() not in ("unspecified", "not specified"):
            norm_val = normalize_match(value)
            in_quote = bool(norm_val and norm_val in evidence_quote)
            in_context = bool(norm_val and normalized_context and norm_val in normalized_context)
            if not (in_quote or in_context):
                validated[field] = "Unspecified"
                validated["status"] = "Needs Review"
    return validated


# Verifies that meeting decisions and actions are supported by transcript evidence and flags proposals.
def validate_record(
    record: dict[str, Any], refined: dict[str, Any]
) -> dict[str, Any]:
    output = {
        "meeting_title": str(record.get("meeting_title") or "Meeting"),
        "summary": str(record.get("summary") or ""),
        "minutes": record.get("minutes") or [],
        "decisions": [],
        "non_decisions": [],
        "action_items": [],
        "discussion_points": record.get("discussion_points") or [],
        "open_questions": record.get("open_questions") or [],
    }
    segments = refined.get("refined_segments") or []
    for decision in record.get("decisions") or []:
        item = dict(decision)
        raw_status = str(item.get("status") or "")
        item["status"] = raw_status if raw_status in ALLOWED_DECISION_STATUSES else "Needs Review"

        item["evidence"] = find_evidence(item.get("evidence_quote", ""), segments)
        if item["evidence"]["found"]:
            evidence_text = item.get("evidence_quote", "")
            decision_text = item.get("text", "")
            if _PROPOSAL_INDICATOR_PATTERN.search(evidence_text) or _PROPOSAL_INDICATOR_PATTERN.search(decision_text):
                item["status"] = "Proposal / Undecided"
                output["non_decisions"].append(item)
            else:
                output["decisions"].append(item)
        else:
            # Never silently discard a decision when evidence cannot be verified;
            # keep it in the final record as "Needs Review" to make validation auditable.
            item["status"] = "Needs Review"
            output["decisions"].append(item)
    for proposal in record.get("non_decisions") or []:
        item = dict(proposal)
        item["evidence"] = find_evidence(item.get("evidence_quote", ""), segments)
        output["non_decisions"].append(item)
    for action in record.get("action_items") or []:
        evidence = find_evidence(action.get("evidence_quote", ""), segments)
        context_text = ""
        if evidence["found"]:
            matched_ids = set(evidence.get("segment_ids", []))
            indices = [i for i, seg in enumerate(segments) if seg.get("id") in matched_ids]
            if indices:
                min_idx = max(0, min(indices) - 2)
                max_idx = min(len(segments), max(indices) + 3)
                context_text = " ".join(seg.get("text", "") for seg in segments[min_idx:max_idx])
        item = validate_action(action, context_text=context_text)
        item["evidence"] = evidence
        if not item["evidence"]["found"]:
            item["status"] = "Needs Review"
            item["owner"] = "Unspecified"
            item["deadline"] = "Unspecified"
        output["action_items"].append(item)
    return output
