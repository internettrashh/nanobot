"""Tests for the TUI activity indicator."""

from nanobot.cli.activity import truncate, format_tool_status, ActivityIndicator


# --- truncate() ---


def test_truncate_short_string():
    assert truncate("hello", 60) == "hello"


def test_truncate_long_path_preserves_tail():
    path = "/very/long/deeply/nested/path/to/some/important_file.py"
    result = truncate(path, 30)
    assert result.startswith("...")
    assert result.endswith("important_file.py")
    assert len(result) <= 30


def test_truncate_long_command_preserves_head():
    cmd = "pip install some-very-long-package-name-that-goes-on-forever --verbose"
    result = truncate(cmd, 30)
    assert result.endswith("...")
    assert result.startswith("pip")
    assert len(result) <= 30


def test_truncate_exact_length():
    s = "a" * 60
    assert truncate(s, 60) == s


# --- format_tool_status() ---


def test_format_read_file():
    assert format_tool_status("read_file", {"path": "/foo/bar.py"}) == "Reading /foo/bar.py"


def test_format_exec():
    assert format_tool_status("exec", {"command": "git status"}) == "Running `git status`"


def test_format_web_search():
    assert format_tool_status("web_search", {"query": "python asyncio"}) == "Searching python asyncio"


def test_format_unknown_tool():
    assert format_tool_status("unknown_tool", {"x": 1}) == "unknown_tool"


def test_format_message_no_key_arg():
    assert format_tool_status("message", {"content": "hello"}) == "Sending message"


def test_format_write_file():
    assert format_tool_status("write_file", {"path": "/tmp/out.txt", "content": "data"}) == "Writing /tmp/out.txt"


# --- ActivityIndicator callback ---


def test_callback_thinking_step_1():
    from unittest.mock import MagicMock
    from rich.console import Console

    console = Console(file=MagicMock())
    indicator = ActivityIndicator(console)
    cb = indicator.get_callback()

    cb("thinking", "", {"iteration": "1"})
    assert indicator._current_text == "nanobot is thinking..."


def test_callback_thinking_step_n():
    from unittest.mock import MagicMock
    from rich.console import Console

    console = Console(file=MagicMock())
    indicator = ActivityIndicator(console)
    cb = indicator.get_callback()

    cb("thinking", "", {"iteration": "3"})
    assert "step 3" in indicator._current_text


def test_callback_tool_start():
    from unittest.mock import MagicMock
    from rich.console import Console

    console = Console(file=MagicMock())
    indicator = ActivityIndicator(console)
    cb = indicator.get_callback()

    cb("tool_start", "read_file", {"path": "/foo.py"})
    assert "Reading" in indicator._current_text
    assert "/foo.py" in indicator._current_text


def test_callback_tool_end_resets():
    from unittest.mock import MagicMock
    from rich.console import Console

    console = Console(file=MagicMock())
    indicator = ActivityIndicator(console)
    cb = indicator.get_callback()

    cb("tool_start", "exec", {"command": "ls"})
    cb("tool_end", "exec")
    assert indicator._current_text == "nanobot is thinking..."


def test_callback_done():
    from unittest.mock import MagicMock
    from rich.console import Console

    console = Console(file=MagicMock())
    indicator = ActivityIndicator(console)
    cb = indicator.get_callback()

    cb("done")
    assert indicator._current_text == "done"
