"""Tests for the subagent tracking system."""

from datetime import datetime, timedelta

from nanobot.agent.subagent import SubagentInfo, SubagentTracker


# ---------------------------------------------------------------------------
# SubagentInfo
# ---------------------------------------------------------------------------


def test_info_elapsed_running():
    info = SubagentInfo(task_id="abc", task="do stuff", label="stuff", origin={})
    assert info.elapsed_seconds >= 0
    assert info.elapsed_seconds < 1


def test_info_elapsed_completed():
    info = SubagentInfo(task_id="abc", task="do stuff", label="stuff", origin={})
    info.started_at = datetime.now() - timedelta(seconds=5)
    info.ended_at = datetime.now()
    assert 4.5 < info.elapsed_seconds < 6.0


def test_display_status_thinking():
    info = SubagentInfo(task_id="abc", task="t", label="l", origin={})
    info.current_phase = "thinking"
    info.iteration = 3
    assert "thinking" in info.display_status
    assert "3/15" in info.display_status


def test_display_status_tool_running():
    info = SubagentInfo(task_id="abc", task="t", label="l", origin={})
    info.current_phase = "tool_running"
    info.current_tool = "read_file"
    info.current_tool_args = {"path": "/foo.py"}
    info.iteration = 2
    assert "Reading" in info.display_status
    assert "2/15" in info.display_status


def test_display_status_completed():
    info = SubagentInfo(task_id="abc", task="t", label="l", origin={})
    info.status = "completed"
    assert "completed" in info.display_status


def test_display_status_starting():
    info = SubagentInfo(task_id="abc", task="t", label="l", origin={})
    assert "starting" in info.display_status


# ---------------------------------------------------------------------------
# SubagentTracker
# ---------------------------------------------------------------------------


def test_register_and_get():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    assert tracker.get("a1") is info
    assert tracker.get("nonexistent") is None


def test_update_phase_thinking():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    tracker.update_phase("a1", "thinking", iteration=2)
    assert info.current_phase == "thinking"
    assert info.iteration == 2
    assert info.current_tool is None


def test_update_phase_tool_running():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    tracker.update_phase(
        "a1", "tool_running",
        iteration=1, tool_name="exec", tool_args={"command": "ls"},
    )
    assert info.current_tool == "exec"
    assert info.current_tool_args == {"command": "ls"}
    assert info.tools_used == ["exec"]


def test_update_phase_accumulates_tools():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    tracker.update_phase("a1", "tool_running", tool_name="read_file", tool_args={})
    tracker.update_phase("a1", "tool_running", tool_name="exec", tool_args={})
    assert info.tools_used == ["read_file", "exec"]


def test_update_phase_nonexistent_is_noop():
    tracker = SubagentTracker()
    tracker.update_phase("nope", "thinking", iteration=1)  # should not raise


def test_mark_completed():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    tracker.mark_completed("a1", "completed", result_summary="done")
    assert info.status == "completed"
    assert info.current_phase == "done"
    assert info.ended_at is not None
    assert info.result_summary == "done"
    assert info.current_tool is None


def test_mark_error():
    tracker = SubagentTracker()
    info = SubagentInfo(task_id="a1", task="t", label="l", origin={})
    tracker.register(info)
    tracker.mark_completed("a1", "error", error_message="boom")
    assert info.status == "error"
    assert info.error_message == "boom"


def test_get_running_filters_completed():
    tracker = SubagentTracker()
    a1 = SubagentInfo(task_id="a1", task="t", label="l1", origin={})
    a2 = SubagentInfo(task_id="a2", task="t", label="l2", origin={})
    tracker.register(a1)
    tracker.register(a2)
    tracker.mark_completed("a1", "completed")
    running = tracker.get_running()
    assert len(running) == 1
    assert running[0].task_id == "a2"


def test_get_all_includes_completed():
    tracker = SubagentTracker()
    a1 = SubagentInfo(task_id="a1", task="t", label="l1", origin={})
    a2 = SubagentInfo(task_id="a2", task="t", label="l2", origin={})
    tracker.register(a1)
    tracker.register(a2)
    tracker.mark_completed("a1", "completed")
    assert len(tracker.get_all()) == 2


def test_get_running_count():
    tracker = SubagentTracker()
    a1 = SubagentInfo(task_id="a1", task="t", label="l1", origin={})
    a2 = SubagentInfo(task_id="a2", task="t", label="l2", origin={})
    tracker.register(a1)
    tracker.register(a2)
    assert tracker.get_running_count() == 2
    tracker.mark_completed("a1", "completed")
    assert tracker.get_running_count() == 1


def test_prune_completed():
    tracker = SubagentTracker(max_completed=2)
    for i in range(5):
        info = SubagentInfo(task_id=f"a{i}", task="t", label=f"l{i}", origin={})
        tracker.register(info)
        tracker.mark_completed(f"a{i}", "completed")
    all_agents = tracker.get_all()
    assert len(all_agents) == 2


def test_prune_keeps_running():
    tracker = SubagentTracker(max_completed=1)
    running = SubagentInfo(task_id="r1", task="t", label="running", origin={})
    tracker.register(running)
    for i in range(3):
        info = SubagentInfo(task_id=f"c{i}", task="t", label=f"done{i}", origin={})
        tracker.register(info)
        tracker.mark_completed(f"c{i}", "completed")
    # Running agent should never be pruned
    assert tracker.get("r1") is not None
    assert tracker.get("r1").status == "running"


# ---------------------------------------------------------------------------
# SubagentStatusTool
# ---------------------------------------------------------------------------


import pytest


@pytest.fixture
def populated_tracker():
    tracker = SubagentTracker()
    a1 = SubagentInfo(task_id="abc12345", task="Analyze the code", label="analyze", origin={})
    a1.current_phase = "thinking"
    a1.iteration = 3
    tracker.register(a1)

    a2 = SubagentInfo(task_id="def67890", task="Run the tests", label="tests", origin={})
    tracker.register(a2)
    tracker.mark_completed("def67890", "completed", result_summary="All tests pass")

    return tracker


@pytest.mark.asyncio
async def test_status_tool_list(populated_tracker):
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tool = SubagentStatusTool(tracker=populated_tracker)
    result = await tool.execute(action="list")
    assert "1 running" in result
    assert "abc12345" in result
    assert "analyze" in result


@pytest.mark.asyncio
async def test_status_tool_list_empty():
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tracker = SubagentTracker()
    tool = SubagentStatusTool(tracker=tracker)
    result = await tool.execute(action="list")
    assert "No subagents" in result


@pytest.mark.asyncio
async def test_status_tool_detail(populated_tracker):
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tool = SubagentStatusTool(tracker=populated_tracker)
    result = await tool.execute(action="detail", task_id="abc12345")
    assert "abc12345" in result
    assert "analyze" in result
    assert "Analyze the code" in result
    assert "3/15" in result


@pytest.mark.asyncio
async def test_status_tool_detail_missing():
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tracker = SubagentTracker()
    tool = SubagentStatusTool(tracker=tracker)
    result = await tool.execute(action="detail", task_id="nope")
    assert "No subagent found" in result


@pytest.mark.asyncio
async def test_status_tool_detail_no_id():
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tracker = SubagentTracker()
    tool = SubagentStatusTool(tracker=tracker)
    result = await tool.execute(action="detail")
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_status_tool_all(populated_tracker):
    from nanobot.agent.tools.subagent_status import SubagentStatusTool
    tool = SubagentStatusTool(tracker=populated_tracker)
    result = await tool.execute(action="all")
    assert "Running" in result
    assert "Completed" in result
    assert "abc12345" in result
    assert "def67890" in result
