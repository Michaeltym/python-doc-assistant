"""HTTP service layer for python-doc-assistant.

Public exports:
    AskRequest, AskState — request schema and shared app state.
    build_app             — FastAPI app constructor.
    token_event, done_event, error_event — SSE event helpers.
    EVENT_TOKEN, EVENT_DONE, EVENT_ERROR — event names.
"""

from python_doc_assistant.service.app import AskRequest, AskState, build_app
from python_doc_assistant.service.streaming import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_TOKEN,
    done_event,
    error_event,
    token_event,
)

__all__ = [
    "AskRequest",
    "AskState",
    "EVENT_DONE",
    "EVENT_ERROR",
    "EVENT_TOKEN",
    "build_app",
    "done_event",
    "error_event",
    "token_event",
]
