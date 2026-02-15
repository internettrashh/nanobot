"""Tests for supermemory integration in MemoryStore."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.memory import MemoryStore


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with memory directory."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# No supermemory (backward compatibility)
# ---------------------------------------------------------------------------


def test_no_supermemory_fallback(workspace):
    """Without API key, MemoryStore behaves exactly as before."""
    store = MemoryStore(workspace)
    assert not store.has_supermemory
    assert store.search("anything") == []
    assert store.get_memory_context() == ""


def test_read_write_long_term_without_supermemory(workspace):
    store = MemoryStore(workspace)
    store.write_long_term("user likes coffee")
    assert store.read_long_term() == "user likes coffee"
    assert "Long-term Memory" in store.get_memory_context()


def test_append_history_without_supermemory(workspace):
    store = MemoryStore(workspace)
    store.append_history("[2026-02-15] user asked about weather")
    content = store.history_file.read_text(encoding="utf-8")
    assert "weather" in content


# ---------------------------------------------------------------------------
# With supermemory (mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sm_client():
    """Create a mocked Supermemory client."""
    client = MagicMock()
    client.add = MagicMock()

    # Mock search results
    result_obj = MagicMock()
    result_obj.content = "user prefers dark mode"
    client.search.documents = MagicMock(
        return_value=MagicMock(results=[result_obj])
    )
    return client


@pytest.fixture
def store_with_sm(workspace, mock_sm_client):
    """MemoryStore with a mocked supermemory client injected."""
    store = MemoryStore(workspace)
    store._sm_client = mock_sm_client
    store._container_tag = "test"
    return store


def test_has_supermemory(store_with_sm):
    assert store_with_sm.has_supermemory


def test_write_long_term_syncs_to_supermemory(store_with_sm, mock_sm_client):
    store_with_sm.write_long_term("important fact")
    mock_sm_client.add.assert_called_once_with(
        content="important fact",
        container_tags=["test"],
        metadata={"type": "long_term_memory"},
    )


def test_append_history_syncs_to_supermemory(store_with_sm, mock_sm_client):
    store_with_sm.append_history("[2026-02-15] did a thing")
    mock_sm_client.add.assert_called_once_with(
        content="[2026-02-15] did a thing",
        container_tags=["test"],
        metadata={"type": "history"},
    )


def test_search_returns_results(store_with_sm):
    results = store_with_sm.search("dark mode")
    assert results == ["user prefers dark mode"]


def test_search_failure_returns_empty(store_with_sm, mock_sm_client):
    mock_sm_client.search.documents.side_effect = Exception("network error")
    results = store_with_sm.search("anything")
    assert results == []


def test_sm_add_failure_does_not_raise(store_with_sm, mock_sm_client):
    mock_sm_client.add.side_effect = Exception("network error")
    # Should not raise
    store_with_sm.write_long_term("data")
    # File should still be written locally
    assert store_with_sm.read_long_term() == "data"


def test_search_respects_limit(workspace, mock_sm_client):
    results = [MagicMock(content=f"memory {i}") for i in range(10)]
    mock_sm_client.search.documents.return_value = MagicMock(results=results)

    store = MemoryStore(workspace)
    store._sm_client = mock_sm_client
    store._container_tag = "test"

    found = store.search("query", limit=3)
    assert len(found) == 3


def test_search_extracts_from_chunks(workspace):
    """When result.content is None, extract from result.chunks."""
    client = MagicMock()
    chunk = MagicMock()
    chunk.content = "recalled from chunks"
    result_obj = MagicMock()
    result_obj.content = None  # top-level is None
    result_obj.chunks = [chunk]
    client.search.documents.return_value = MagicMock(results=[result_obj])

    store = MemoryStore(workspace)
    store._sm_client = client
    store._container_tag = "test"

    found = store.search("query")
    assert found == ["recalled from chunks"]
