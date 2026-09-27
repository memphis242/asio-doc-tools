import json
from hashlib import sha256

from classify_support import entry, good_item, text_response

from asio_doc_tools import classify
from asio_doc_tools.classify import prompt


def test_the_store_key_of_a_fixed_text_never_changes() -> None:
    # Every stored (paid-for) classification is found by this key: a change to the prompt
    # version, model, effort, or the key's formula would make every one a miss.
    assert classify.cache_key(entry("Fixed a bug.")) == (
        "e1e6bf83860ec8706dfb88c18fa9645d1bdf44f1f5e456f0b26ad7330730addc"
    )
    assert (classify.PROMPT_VERSION, classify.MODEL, classify.EFFORT) == (
        "classify-2",
        "claude-sonnet-5",
        "low",
    )


def test_the_prompt_and_schema_are_unchanged() -> None:
    assert sha256(prompt.SYSTEM_PROMPT.encode()).hexdigest() == (
        "2ecd07ccc72af20be817189e883d4b73fb7a4d24751dee7f6223fd3190fd9e26"
    )
    assert sha256(json.dumps(prompt.SCHEMA).encode()).hexdigest() == (
        "280f344e10bfd327f6ab08f96747e8b3d510643845f507cb94e456e9b9ea0757"
    )


def test_a_row_key_matches_the_cache_key_under_this_prompt() -> None:
    text = "Changed a default.\n  - with a sub-point"
    assert prompt.row_key(prompt.PROMPT_VERSION, prompt.MODEL, prompt.EFFORT, text) == prompt.key_for_text(
        text
    )
    assert prompt.row_key("classify-1", prompt.MODEL, prompt.EFFORT, text) != prompt.key_for_text(text)


def test_the_request_has_no_model_choice_or_fallback() -> None:
    kwargs = prompt.request_kwargs(2_176, "{}")
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert set(kwargs) == {"model", "max_tokens", "thinking", "output_config", "system", "messages"}
    assert "cache_control" not in json.dumps(kwargs)


def test_entries_within_the_cap_are_sent_unchanged_and_longer_ones_are_cut_visibly() -> None:
    within = "y" * prompt.ENTRY_TEXT_CAP
    assert prompt.payload_text(within) == within
    longer = "z" * (prompt.ENTRY_TEXT_CAP + 500)
    sent = prompt.payload_text(longer)
    assert sent.startswith(longer[: prompt.ENTRY_TEXT_CAP]) and "500 more characters truncated" in sent


def test_a_reply_is_valid_only_with_exactly_the_ids_sent() -> None:
    ids = ("e0", "e1")
    assert prompt.validate_reply(text_response([good_item("e0"), good_item("e1")]), ids) is not None
    assert prompt.validate_reply(text_response([good_item("e0")]), ids) is None
    assert prompt.validate_reply(text_response([good_item("e0"), good_item("e0")]), ids) is None
    assert "missing ids ['e1']" in prompt.diagnose_reply(text_response([good_item("e0")]), ids)
