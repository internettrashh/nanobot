"""Tests for supermemory integration in MemoryStore."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.filesystem import WriteFileTool, EditFileTool


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


# ---------------------------------------------------------------------------
# sync_to_cloud
# ---------------------------------------------------------------------------


def test_sync_to_cloud_calls_sm_add(store_with_sm, mock_sm_client):
    store_with_sm.sync_to_cloud("cloud only content", "long_term")
    mock_sm_client.add.assert_called_once_with(
        content="cloud only content",
        container_tags=["test"],
        metadata={"type": "long_term_memory"},
    )


def test_sync_to_cloud_noop_without_supermemory(workspace):
    store = MemoryStore(workspace)
    # Should not raise
    store.sync_to_cloud("content", "history")


# ---------------------------------------------------------------------------
# File tool → supermemory sync via on_memory_write callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_tool_syncs_memory_md(workspace):
    """WriteFileTool fires on_memory_write when writing MEMORY.md."""
    calls = []
    def on_write(mem_type, content):
        calls.append((mem_type, content))

    tool = WriteFileTool(on_memory_write=on_write)
    mem_path = str(workspace / "memory" / "MEMORY.md")
    await tool.execute(path=mem_path, content="user likes tea")
    assert len(calls) == 1
    assert calls[0] == ("long_term", "user likes tea")


@pytest.mark.asyncio
async def test_write_file_tool_syncs_history_md(workspace):
    """WriteFileTool fires on_memory_write when writing HISTORY.md."""
    calls = []
    def on_write(mem_type, content):
        calls.append((mem_type, content))

    tool = WriteFileTool(on_memory_write=on_write)
    hist_path = str(workspace / "memory" / "HISTORY.md")
    await tool.execute(path=hist_path, content="[2026-02-15] event")
    assert len(calls) == 1
    assert calls[0] == ("history", "[2026-02-15] event")


@pytest.mark.asyncio
async def test_write_file_tool_no_callback_for_other_files(workspace):
    """WriteFileTool does NOT fire on_memory_write for non-memory files."""
    calls = []
    def on_write(mem_type, content):
        calls.append((mem_type, content))

    tool = WriteFileTool(on_memory_write=on_write)
    other_path = str(workspace / "notes.txt")
    await tool.execute(path=other_path, content="random")
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_edit_file_tool_syncs_memory_md(workspace):
    """EditFileTool fires on_memory_write when editing MEMORY.md."""
    # Write initial content
    mem_file = workspace / "memory" / "MEMORY.md"
    mem_file.write_text("old fact", encoding="utf-8")

    calls = []
    def on_write(mem_type, content):
        calls.append((mem_type, content))

    tool = EditFileTool(on_memory_write=on_write)
    await tool.execute(path=str(mem_file), old_text="old fact", new_text="new fact")
    assert len(calls) == 1
    assert calls[0] == ("long_term", "new fact")


@pytest.mark.asyncio
async def test_edit_file_tool_no_callback_for_other_files(workspace):
    """EditFileTool does NOT fire on_memory_write for non-memory files."""
    other_file = workspace / "data.txt"
    other_file.write_text("hello world", encoding="utf-8")

    calls = []
    def on_write(mem_type, content):
        calls.append((mem_type, content))

    tool = EditFileTool(on_memory_write=on_write)
    await tool.execute(path=str(other_file), old_text="hello", new_text="goodbye")
    assert len(calls) == 0
