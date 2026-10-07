from app.pipeline.evidence import protected_tokens, safe_edit


def test_protected_numbers_and_monetary_values_cannot_change():
    assert not safe_edit("budget is ₹2.5 lakh", "budget is ₹25 lakh")[0]
    assert not safe_edit("budget is 2.5 lakh", "budget is 2.5 crore")[0]
    assert not safe_edit("budget is $500", "budget is 500")[0]
    assert not safe_edit("budget is $500", "budget is €500")[0]
    assert not safe_edit("revenue is 12.5%", "revenue is 15.2%")[0]
    assert protected_tokens("cost is INR 1,25,000") == protected_tokens(
        "cost is INR 1,25,000"
    )


def test_negation_cannot_be_removed():
    assert not safe_edit("we will not deploy Friday", "we will deploy Friday")[0]


def test_modality_cannot_be_strengthened():
    assert not safe_edit("we might deploy Friday", "we will deploy Friday")[0]
    assert not safe_edit("we committed to deploy", "we planned to deploy")[0]


def test_safe_terminology_edit_preserves_protected_tokens():
    assert safe_edit("use kuberneties by 2027", "use Kubernetes by 2027")[0]


def test_numbers_2_5_lakh_vs_25_lakh_regression():
    source = "The current estimate is 2.5 lakh rupees, not 25 lakh."
    modified = "The budget is 25 lakh rupees."
    safe, reasons = safe_edit(source, modified)
    assert not safe
    assert "protected information changed" in reasons


def test_negation_will_not_deploy_regression():
    source = "We will not deploy the new model this Friday."
    modified = "The team will deploy the new model this Friday."
    safe, reasons = safe_edit(source, modified)
    assert not safe
    assert "protected information changed" in reasons


def test_existing_names_are_preserved():
    source = "Asha will send the report to Rahul."
    modified = "Asha will send the report to Rahul."
    safe, reasons = safe_edit(source, modified)
    assert safe
    assert reasons == []


def test_unsupported_name_expansion_is_rejected():
    source = "Sue will take care of deployment."
    modified = "Sue Carpenter will take care of deployment."
    safe, reasons = safe_edit(source, modified)
    assert not safe
    assert "unsupported name change" in reasons


def test_unsupported_name_deletion_is_rejected():
    source = "Mark Robert confirmed the release date."
    modified = "Mark confirmed the release date."
    safe, reasons = safe_edit(source, modified)
    assert not safe
    assert "unsupported name change" in reasons


def test_glossary_cannot_expand_unspoken_names():
    # Expanding a spoken first name to full name via glossary is rejected
    source = "Sue will take care of deployment."
    modified = "Sue Carpenter will take care of deployment."
    safe, reasons = safe_edit(source, modified, glossary=["Sue Carpenter"])
    assert not safe
    assert "unsupported name change" in reasons

    source2 = "Jason Somerville did it."
    modified2 = "Jason Somerville did it."
    safe2, reasons2 = safe_edit(source2, modified2, glossary=["Jason Somerville"])
    assert safe2
    assert reasons2 == []

    source3 = "Fine, Jason, you did it."
    modified3 = "Fine, Jason Somerville, you did it."
    safe3, reasons3 = safe_edit(source3, modified3, glossary=["Jason Somerville"])
    assert not safe3
    assert "unsupported name change" in reasons3


def test_number_words_protection_both_directions():
    # twelve -> twenty and twenty -> twelve
    safe1, reasons1 = safe_edit("we have twelve instances", "we have twenty instances")
    assert not safe1
    assert "protected information changed" in reasons1

    safe2, reasons2 = safe_edit("we have twenty instances", "we have twelve instances")
    assert not safe2
    assert "protected information changed" in reasons2

    # fifteen -> fifty and fifty -> fifteen
    safe3, reasons3 = safe_edit("latency is fifteen ms", "latency is fifty ms")
    assert not safe3
    assert "protected information changed" in reasons3

    safe4, reasons4 = safe_edit("latency is fifty ms", "latency is fifteen ms")
    assert not safe4
    assert "protected information changed" in reasons4


def test_technical_terms_allowed_when_source_lacks_person_name():
    # cooper netties -> Kubernetes
    safe1, reasons1 = safe_edit("deploy on cooper netties", "deploy on Kubernetes")
    assert safe1
    assert reasons1 == []

    # post gress -> Postgres
    safe2, reasons2 = safe_edit("migrate to post gress", "migrate to Postgres")
    assert safe2
    assert reasons2 == []


def test_semantic_paraphrasing_is_rejected():
    # smiley fries -> smiley face potatoes
    safe1, reasons1 = safe_edit("they ordered smiley fries", "they ordered smiley face potatoes")
    assert not safe1
    assert "semantic paraphrasing rejected" in reasons1

    # big problem -> major issue
    safe2, reasons2 = safe_edit("this is a big problem", "this is a major issue")
    assert not safe2
    assert "semantic paraphrasing rejected" in reasons2

    # Technical correction remains allowed
    safe3, reasons3 = safe_edit("use post gress for the database", "use Postgres for the database")
    assert safe3
    assert reasons3 == []
