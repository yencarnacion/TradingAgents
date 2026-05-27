from __future__ import annotations

from typing import Any, Iterable, Sequence


MessageSignature = tuple[Any, ...]


def _message_signature(message: Any) -> MessageSignature:
    """Build a stable signature for streamed LangGraph messages.

    Prefer the message id when present; otherwise fall back to a tuple of the
    fields that make repeated pretty_print() calls user-visible.
    """

    message_id = getattr(message, "id", None)
    if message_id:
        return ("id", message_id)

    return (
        type(message).__name__,
        getattr(message, "name", None),
        getattr(message, "content", None),
        repr(getattr(message, "tool_calls", None)),
        repr(getattr(message, "additional_kwargs", None)),
        repr(getattr(message, "response_metadata", None)),
    )


def split_new_messages(messages: Sequence[Any], seen_signatures: Sequence[MessageSignature]) -> tuple[list[Any], list[MessageSignature]]:
    """Return only the newly appended messages from a streamed cumulative state.

    LangGraph debug streaming surfaces cumulative ``messages`` state on each
    chunk, so naïvely pretty-printing ``messages[-1]`` for every chunk repeats
    the same agent output on later non-message updates. This helper treats the
    common case as append-only growth and degrades gracefully if the message
    list is replaced or truncated.
    """

    current_signatures = [_message_signature(message) for message in messages]

    prefix_len = 0
    max_prefix = min(len(current_signatures), len(seen_signatures))
    while prefix_len < max_prefix and current_signatures[prefix_len] == seen_signatures[prefix_len]:
        prefix_len += 1

    if prefix_len == len(seen_signatures):
        return list(messages[prefix_len:]), current_signatures

    if not messages:
        return [], current_signatures

    if not seen_signatures or current_signatures[-1] != seen_signatures[-1]:
        return [messages[-1]], current_signatures

    return [], current_signatures
