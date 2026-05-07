"""Tests for `python_doc_assistant.service.streaming`."""

from __future__ import annotations

import json

from python_doc_assistant.service.streaming import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_TOKEN,
    _decode,
    done_event,
    error_event,
    token_event,
)

# ------------------------------------------------------------------
# Constant surface
# ------------------------------------------------------------------


def test_event_name_constants_are_strings() -> None:
    assert EVENT_TOKEN == "token"
    assert EVENT_DONE == "done"
    assert EVENT_ERROR == "error"


# ------------------------------------------------------------------
# token_event
# ------------------------------------------------------------------


def test_token_event_event_name_is_token() -> None:
    assert token_event("hello")["event"] == EVENT_TOKEN


def test_token_event_data_is_json_with_text_field() -> None:
    ev = token_event("hello world")
    payload = json.loads(ev["data"])
    assert payload == {"text": "hello world"}


def test_token_event_handles_empty_string() -> None:
    ev = token_event("")
    assert json.loads(ev["data"]) == {"text": ""}


def test_token_event_escapes_special_chars_via_json() -> None:
    text = 'newline\nquote"backslash\\'
    ev = token_event(text)
    # Round-trip through JSON must preserve the original text.
    assert json.loads(ev["data"])["text"] == text


# ------------------------------------------------------------------
# done_event
# ------------------------------------------------------------------


def test_done_event_event_name_is_done() -> None:
    ev = done_event(refused=False, cited_chunks=(), latency_seconds=1.0)
    assert ev["event"] == EVENT_DONE


def test_done_event_payload_carries_required_fields() -> None:
    ev = done_event(
        refused=False,
        cited_chunks=(
            {
                "chunk_id": "symbol:json.loads",
                "title": "json.loads",
                "url": "https://docs.python.org/3.12/library/json.html#json.loads",
            },
            {
                "chunk_id": "symbol:json.dumps",
                "title": "json.dumps",
                "url": "https://docs.python.org/3.12/library/json.html#json.dumps",
            },
        ),
        latency_seconds=12.345,
        model="qwen-7b-gguf",
    )
    payload = json.loads(ev["data"])
    assert payload["refused"] is False
    assert len(payload["cited_chunks"]) == 2
    first = payload["cited_chunks"][0]
    assert first["chunk_id"] == "symbol:json.loads"
    assert first["title"] == "json.loads"
    assert first["url"].startswith("https://docs.python.org/")
    assert payload["latency_seconds"] == 12.345
    assert payload["rewritten_query"] is None
    assert payload["model"] == "qwen-7b-gguf"


def test_done_event_includes_rewritten_query_when_provided() -> None:
    ev = done_event(
        refused=False,
        cited_chunks=(),
        latency_seconds=0.5,
        rewritten_query="json.loads",
    )
    assert json.loads(ev["data"])["rewritten_query"] == "json.loads"


def test_done_event_refused_true_zero_citations() -> None:
    ev = done_event(refused=True, cited_chunks=(), latency_seconds=0.1)
    payload = json.loads(ev["data"])
    assert payload["refused"] is True
    assert payload["cited_chunks"] == []


def test_done_event_carries_trace_fields() -> None:
    ev = done_event(
        refused=False,
        cited_chunks=(),
        latency_seconds=0.5,
        query_type="identifier",
        retrieved=(
            {
                "chunk_id": "symbol:json.loads",
                "rank": 1,
                "score": 12.5,
                "title": "json.loads",
                "url": "https://docs.python.org/3.12/library/json.html#json.loads",
                "cited": True,
            },
            {
                "chunk_id": "symbol:json.dumps",
                "rank": 2,
                "score": 9.3,
                "title": "json.dumps",
                "url": "https://docs.python.org/3.12/library/json.html#json.dumps",
                "cited": False,
            },
        ),
    )
    payload = json.loads(ev["data"])
    assert payload["query_type"] == "identifier"
    assert len(payload["retrieved"]) == 2
    assert payload["retrieved"][0]["cited"] is True
    assert payload["retrieved"][1]["cited"] is False
    assert payload["retrieved"][0]["rank"] == 1


def test_done_event_trace_fields_default_to_empty() -> None:
    ev = done_event(refused=False, cited_chunks=(), latency_seconds=0.1)
    payload = json.loads(ev["data"])
    assert payload["query_type"] is None
    assert payload["retrieved"] == []
    assert payload["prompt_messages"] == []


def test_done_event_carries_prompt_messages() -> None:
    ev = done_event(
        refused=False,
        cited_chunks=(),
        latency_seconds=0.1,
        prompt_messages=(
            {"role": "system", "content": "You are a Python docs assistant."},
            {"role": "user", "content": "[1] json.loads ...\n\nQuestion: what is json.loads"},
        ),
    )
    payload = json.loads(ev["data"])
    assert len(payload["prompt_messages"]) == 2
    assert payload["prompt_messages"][0]["role"] == "system"
    assert payload["prompt_messages"][1]["role"] == "user"
    assert "json.loads" in payload["prompt_messages"][1]["content"]


# ------------------------------------------------------------------
# error_event
# ------------------------------------------------------------------


def test_error_event_event_name_is_error() -> None:
    assert error_event("oops")["event"] == EVENT_ERROR


def test_error_event_payload_has_message() -> None:
    ev = error_event("retrieval failed")
    assert json.loads(ev["data"]) == {"message": "retrieval failed"}


# ------------------------------------------------------------------
# _decode helper
# ------------------------------------------------------------------


def test_decode_returns_event_name_and_payload() -> None:
    name, payload = _decode(token_event("hi"))
    assert name == EVENT_TOKEN
    assert payload == {"text": "hi"}
