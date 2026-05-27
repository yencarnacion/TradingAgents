from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tradingagents" / "graph" / "debug_stream.py"
_SPEC = spec_from_file_location("tradingagents_graph_debug_stream", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DEBUG_STREAM = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEBUG_STREAM)
split_new_messages = _DEBUG_STREAM.split_new_messages


class FakeMessage:
    def __init__(self, content, message_id=None):
        self.content = content
        self.id = message_id


def test_split_new_messages_only_returns_appended_entries_for_cumulative_streams():
    seen = []

    first = [FakeMessage("research", "m1")]
    new_messages, seen = split_new_messages(first, seen)
    assert [message.content for message in new_messages] == ["research"]

    repeated_chunk = [FakeMessage("research", "m1")]
    new_messages, seen = split_new_messages(repeated_chunk, seen)
    assert new_messages == []

    appended_chunk = [FakeMessage("research", "m1"), FakeMessage("decision", "m2")]
    new_messages, seen = split_new_messages(appended_chunk, seen)
    assert [message.content for message in new_messages] == ["decision"]

    repeated_appended_chunk = [FakeMessage("research", "m1"), FakeMessage("decision", "m2")]
    new_messages, seen = split_new_messages(repeated_appended_chunk, seen)
    assert new_messages == []


def test_split_new_messages_falls_back_to_last_message_when_stream_state_is_replaced():
    original = [FakeMessage("old", "m1")]
    _, seen = split_new_messages(original, [])

    replaced = [FakeMessage("new", "m2")]
    new_messages, seen = split_new_messages(replaced, seen)

    assert [message.content for message in new_messages] == ["new"]
    assert seen == [("id", "m2")]
