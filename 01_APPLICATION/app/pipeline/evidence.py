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


# Common non-name words that may appear capitalized (sentence starters, days, months, common nouns).
_COMMON_NON_NAME_WORDS = {
    "the", "a", "an", "we", "i", "it", "they", "this", "that", "these", "those",
    "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "our", "my", "your", "his", "her",
    "its", "their", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "ok", "okay", "yes", "yeah", "sure", "hello", "hi", "hey", "bye", "good",
    "morning", "afternoon", "evening", "thanks", "thank", "please", "meeting",
    "today", "tomorrow", "yesterday", "next", "last", "first", "second",
    "budget", "cost", "revenue", "estimate", "team", "launch", "model",
}


def _extract_name_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z][a-z]+\b", str(text))
    return [t for t in tokens if t.lower() not in _COMMON_NON_NAME_WORDS]


# Extracts and normalizes protected factual tokens (numbers, currencies, dates, negations) from text.
def protected_tokens(text: str) -> list[str]:
    return sorted(
        match.group(0).lower().replace(" ", "")
        for match in _PROTECTED_TOKEN_PATTERN.finditer(str(text))
    )


# Ensures proposed transcript edits do not modify, remove, or corrupt protected factual information or names.
def safe_edit(
    source: str, target: str, glossary: list[str] | None = None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if protected_tokens(source) != protected_tokens(target):
        reasons.append("protected information changed")

    # Name protection: prevent unsupported name expansion or deletion
    glossary_tokens: set[str] = set()
    if glossary:
        for g_item in glossary:
            for g_tok in re.findall(r"\b\w+\b", str(g_item).lower()):
                glossary_tokens.add(g_tok)

    src_words = [w.lower() for w in re.findall(r"\b\w+\b", str(source))]
    tgt_words = [w.lower() for w in re.findall(r"\b\w+\b", str(target))]

    src_names = _extract_name_tokens(source)
    tgt_names = _extract_name_tokens(target)

    # 1. Unsupported name expansion: new capitalized name in target that wasn't in source
    for t_name in tgt_names:
        t_low = t_name.lower()
        if not any(
            sw == t_low or difflib.SequenceMatcher(None, sw, t_low).ratio() >= 0.85
            for sw in src_words
        ):
            if t_low not in glossary_tokens:
                reasons.append("unsupported name change")
                break

    # 2. Unsupported name deletion: capitalized name in source deleted from target
    if "unsupported name change" not in reasons:
        for s_name in src_names:
            s_low = s_name.lower()
            if not any(
                tw == s_low or difflib.SequenceMatcher(None, tw, s_low).ratio() >= 0.85
                for tw in tgt_words
            ):
                if s_low not in glossary_tokens:
                    reasons.append("unsupported name change")
                    break

    safe = len(reasons) == 0
    return safe, reasons


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
    r"idea|option|consider|considering|might|could|maybe|perhaps|possibly|possibility|"
    r"estimate|estimates|estimated|estimating|estimation|predict|predicts|predicted|prediction|"
    r"clarify|clarified|clarification|budget estimate|current estimate|"
    r"need to finalize|needs to finalize|need to complete|needs to complete|"
    r"need to decide|needs to decide|still need to|to be decided|"
    r"not made a final decision|have not made a final decision|not yet decided|"
    r"haven't decided|has not decided|undecided|under discussion|review in next|"
    r"review at next|discuss at next|decide at next)\b",
    re.IGNORECASE,
)

# Regex matching words claiming finalized completion or approval
_CLAIM_COMPLETION_PATTERN = re.compile(
    r"\b(?:finalized|completed|approved)\b",
    re.IGNORECASE,
)

# Sanitizes meeting summary against unsupported claims of completion/finalization when work was only discussed or planned.
def sanitize_summary_overclaims(summary: str, refined_text: str) -> str:
    if not summary:
        return ""
    trans_lower = refined_text.lower()
    if "need to finalize" in trans_lower or "needs to finalize" in trans_lower:
        if not re.search(r"\b(?:team\s+finalized|we\s+finalized|have\s+finalized|has\s+finalized)\b", trans_lower):
            summary = re.sub(
                r"\b(?:team\s+)?finalized\s+(?:the\s+)?(deployment\s+(?:schedule|plan))",
                r"discussed the \1",
                summary,
                flags=re.IGNORECASE,
            )
            summary = re.sub(
                r"\bfinalized\s+",
                r"planned ",
                summary,
                flags=re.IGNORECASE,
            )
    return summary


# Allowed status values defined by the meeting documentation schema.
ALLOWED_DECISION_STATUSES = {"Confirmed", "Needs Review"}
ALLOWED_ACTION_STATUSES = {"Confirmed", "Needs Review"}


_FIRST_PERSON_COMMITMENT_PATTERN = re.compile(
    r"\b(?:i'll|i will|i can|i shall|i'm going to|i am going to|"
    r"i've committed|i commit|i'd be happy to|i plan to)\b",
    re.IGNORECASE,
)


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

    # Outstanding task without assigned owner or deadline inferred from need/follow-up defaults to Needs Review
    if validated["owner"] == "Unspecified" and validated["deadline"] == "Unspecified":
        quote_or_task = (validated.get("evidence_quote", "") + " " + validated.get("task", "")).lower()
        if any(term in quote_or_task for term in ("need to", "needs to", "still need", "should", "must", "investigate", "someone", "next meeting", "review")):
            validated["status"] = "Needs Review"

    # Action owner note for first-person commitments without acoustic speaker identification
    if validated["owner"] == "Unspecified":
        quote_text = validated.get("evidence_quote", "")
        task_text = validated.get("task", "")
        if (
            _FIRST_PERSON_COMMITMENT_PATTERN.search(quote_text)
            or _FIRST_PERSON_COMMITMENT_PATTERN.search(task_text)
            or _FIRST_PERSON_COMMITMENT_PATTERN.search(context_text)
        ):
            validated["owner_note"] = "First-person commitment; speaker identity unavailable."

    return validated


# Verifies that meeting decisions and actions are supported by transcript evidence and flags proposals.
def validate_record(
    record: dict[str, Any], refined: dict[str, Any]
) -> dict[str, Any]:
    refined_text = str(refined.get("refined_text") or "")
    output = {
        "meeting_title": str(record.get("meeting_title") or "Meeting"),
        "summary": sanitize_summary_overclaims(str(record.get("summary") or ""), refined_text),
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
            elif (
                _CLAIM_COMPLETION_PATTERN.search(decision_text)
                and not _CLAIM_COMPLETION_PATTERN.search(evidence_text)
            ):
                item["status"] = "Needs Review"
                output["decisions"].append(item)
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

    # Cross-section deduplication: avoid representing the exact same statement redundantly
    # in both decisions and action_items
    deduped_actions: list[dict[str, Any]] = []
    decision_texts = {
        normalize_match(d.get("text", "")): d
        for d in output["decisions"]
    }
    for action in output["action_items"]:
        task_norm = normalize_match(action.get("task", ""))
        matching_dec = decision_texts.get(task_norm)
        if matching_dec:
            # If the action has NO assigned work (owner and deadline both Unspecified),
            # it is an unassigned duplicate statement of the confirmed decision.
            if (
                action.get("owner") == "Unspecified"
                and action.get("deadline") == "Unspecified"
            ):
                continue
        deduped_actions.append(action)
    output["action_items"] = deduped_actions

    return output
