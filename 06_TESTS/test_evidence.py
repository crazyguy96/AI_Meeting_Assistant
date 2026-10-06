from app.pipeline.evidence import find_evidence, validate_action, validate_record


def test_evidence_matches_across_adjacent_segments():
    segments = [
        {"id": "S1", "start": 1.0, "text": "We approved the"},
        {"id": "S2", "start": 2.0, "text": "launch for Friday."},
    ]
    evidence = find_evidence("approved the launch for Friday", segments)
    assert evidence["found"]
    assert evidence["segment_ids"] == ["S1", "S2"]


def test_unsupported_owner_and_deadline_are_removed():
    action = {
        "task": "Send the report",
        "owner": "Asha",
        "deadline": "Friday",
        "evidence_quote": "We should send the report.",
        "status": "Confirmed",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Unspecified"
    assert validated["status"] == "Needs Review"


def test_supported_owner_and_deadline_are_retained():
    action = {
        "task": "Send the report",
        "owner": "Asha",
        "deadline": "Friday",
        "evidence_quote": "Asha will send the report by Friday.",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Asha"
    assert validated["deadline"] == "Friday"


def test_unsupported_decision_is_retained_as_needs_review_and_proposal_is_non_decision():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We could consider a June launch."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Launch in May",
                "evidence_quote": "We decided to launch in May.",
                "status": "Confirmed",
            }
        ],
        "non_decisions": [
            {
                "text": "Consider a June launch",
                "evidence_quote": "We could consider a June launch.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Launch in May"
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["decisions"][0]["evidence"]["found"] is False
    assert len(validated["non_decisions"]) == 1
    assert validated["non_decisions"][0]["text"] == "Consider a June launch"


def test_decision_with_missing_evidence_remains_in_output_as_needs_review():
    refined = {"refined_segments": [{"id": "S1", "start": 0.0, "text": "Unrelated meeting transcript."}]}
    source = {
        "decisions": [
            {
                "text": "Adopt microservices architecture",
                "status": "Confirmed",
                "evidence_quote": "We fully agreed to adopt microservices.",
            }
        ]
    }
    validated = validate_record(source, refined)
    assert len(validated["decisions"]) == 1
    assert validated["decisions"][0]["text"] == "Adopt microservices architecture"
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["decisions"][0]["evidence"]["found"] is False
    assert validated["decisions"][0]["evidence_quote"] == "We fully agreed to adopt microservices."


def test_valid_decision_and_action_statuses_are_preserved():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We approved budget and Rahul will deploy tomorrow."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Approved budget",
                "status": "Confirmed",
                "evidence_quote": "We approved budget",
            }
        ],
        "action_items": [
            {
                "task": "Deploy tomorrow",
                "owner": "Rahul",
                "deadline": "tomorrow",
                "status": "Confirmed",
                "evidence_quote": "Rahul will deploy tomorrow.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert validated["decisions"][0]["status"] == "Confirmed"
    assert validated["action_items"][0]["status"] == "Confirmed"


def test_invalid_llm_status_is_safely_normalized():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 0.0, "text": "We approved budget and Rahul will deploy tomorrow."}
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Approved budget",
                "status": "Finalized_By_Management",
                "evidence_quote": "We approved budget",
            }
        ],
        "action_items": [
            {
                "task": "Deploy tomorrow",
                "owner": "Rahul",
                "deadline": "tomorrow",
                "status": "In_Progress",
                "evidence_quote": "Rahul will deploy tomorrow.",
            }
        ],
    }
    validated = validate_record(source, refined)
    assert validated["decisions"][0]["status"] == "Needs Review"
    assert validated["action_items"][0]["status"] == "Needs Review"


def test_confidently_supported_action_items_not_unnecessarily_marked_needs_review():
    refined = {
        "refined_segments": [
            {"id": "S1", "start": 10.0, "text": "Priya will finalize the API specification by end of day."}
        ]
    }
    source = {
        "action_items": [
            {
                "task": "Finalize the API specification",
                "owner": "Priya",
                "deadline": "end of day",
                "status": "Confirmed",
                "evidence_quote": "Priya will finalize the API specification by end of day.",
            }
        ]
    }
    validated = validate_record(source, refined)
    assert len(validated["action_items"]) == 1
    action = validated["action_items"][0]
    assert action["status"] == "Confirmed"
    assert action["owner"] == "Priya"
    assert action["deadline"] == "end of day"
    assert action["evidence"]["found"] is True


def test_proposal_not_incorrectly_classified_as_decision():
    refined = {
        "refined_segments": [
            {
                "id": "S1",
                "start": 0.0,
                "text": "Rahul suggested that we migrate the database to PostgreSQL 16. We discussed the idea, but we have not made a final decision.",
            }
        ]
    }
    source = {
        "decisions": [
            {
                "text": "Migrate database to PostgreSQL 16",
                "evidence_quote": "Rahul suggested that we migrate the database to PostgreSQL 16.",
            }
        ],
        "non_decisions": [],
    }
    validated = validate_record(source, refined)
    # Must NOT be classified as confirmed decision
    assert validated["decisions"] == []
    # Must be categorized as proposal / non-decision
    assert len(validated["non_decisions"]) == 1
    assert "PostgreSQL 16" in validated["non_decisions"][0]["text"]


def test_unspecified_owner_and_deadline_remain_unspecified():
    action = {
        "task": "Review latency metrics",
        "evidence_quote": "We should review latency metrics next week.",
    }
    validated = validate_action(action)
    assert validated["owner"] == "Unspecified"
    assert validated["deadline"] == "Unspecified"


def test_action_deadline_confirmed_in_adjacent_segment_is_retained():
    segments = [
        {"id": "S1", "start": 0.0, "text": "Can someone update the slide deck?"},
        {"id": "S2", "start": 3.0, "text": "I will handle the slide deck."},
        {"id": "S3", "start": 6.0, "text": "Sounds good, please confirm it by Friday."},
    ]
    source = {
        "action_items": [
            {
                "task": "Handle the slide deck",
                "owner": "Unspecified",
                "deadline": "Friday",
                "status": "Confirmed",
                "evidence_quote": "I will handle the slide deck.",
            }
        ]
    }
    refined = {"refined_segments": segments}
    validated = validate_record(source, refined)
    action = validated["action_items"][0]
    assert action["deadline"] == "Friday"
    assert action["status"] == "Confirmed"


def test_action_owner_spoken_in_adjacent_segment_is_retained():
    segments = [
        {"id": "S1", "start": 0.0, "text": "Alice, will you send the summary?"},
        {"id": "S2", "start": 3.0, "text": "Yes, I will send the summary."},
    ]
    source = {
        "action_items": [
            {
                "task": "Send the summary",
                "owner": "Alice",
                "deadline": "Unspecified",
                "status": "Confirmed",
                "evidence_quote": "Yes, I will send the summary.",
            }
        ]
    }
    refined = {"refined_segments": segments}
    validated = validate_record(source, refined)
    action = validated["action_items"][0]
    assert action["owner"] == "Alice"
    assert action["status"] == "Confirmed"


