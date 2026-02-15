"""TUI activity indicator for the nanobot CLI."""

from __future__ import annotations

from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.text import Text


# Maximum display width for argument values (file paths, commands, etc.)
MAX_ARG_DISPLAY = 60


def truncate(value: str, max_len: int = MAX_ARG_DISPLAY) -> str:
    """Truncate a string with ellipsis.

    For file paths, preserves the tail (most informative part).
    For other strings, preserves the head.
    """
    if len(value) <= max_len:
        return value
    if "/" in value or "\\" in value:
        return "..." + value[-(max_len - 3):]
    return value[: max_len - 3] + "..."


# Map tool names to (human-readable verb, key argument to extract).
TOOL_DISPLAY_MAP: dict[str, tuple[str, str | None]] = {
    "read_file": ("Reading", "path"),
    "write_file": ("Writing", "path"),
    "edit_file": ("Editing", "path"),
    "list_dir": ("Listing", "path"),
    "exec": ("Running", "command"),
    "web_search": ("Searching", "query"),
    "web_fetch": ("Fetching", "url"),
    "message": ("Sending message", None),
    "spawn": ("Spawning subagent", "label"),
    "subagent_status": ("Checking subagents", "action"),
    "cron": ("Scheduling", "action"),
}


def format_tool_status(tool_name: str, arguments: dict) -> str:
    """Format a human-readable status string for a tool call.

    Examples::

        format_tool_status("read_file", {"path": "/foo/bar.py"})
        # -> "Reading /foo/bar.py"

        format_tool_status("exec", {"command": "git status"})
        # -> "Running `git status`"
    """
    verb, key_arg = TOOL_DISPLAY_MAP.get(tool_name, (tool_name, None))

    if key_arg and key_arg in arguments:
        arg_value = str(arguments[key_arg])
        arg_display = truncate(arg_value)
        if key_arg == "command":
            return f"{verb} `{arg_display}`"
        return f"{verb} {arg_display}"

    return verb


class ActivityIndicator:
    """Rich Live display showing real-time agent activity.

    Usage::

        indicator = ActivityIndicator(console)
        with indicator:
            callback = indicator.get_callback()
            await agent_loop.process_direct(msg, status_callback=callback)
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None
        self._current_text = "nanobot is thinking..."

    def __enter__(self) -> "ActivityIndicator":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def _render(self) -> Text:
        return Text.from_markup(f"[dim]  {self._current_text}[/dim]")

    def update(self, status: str) -> None:
        """Update the displayed status text."""
        self._current_text = status
        if self._live:
            self._live.update(self._render())

    def get_callback(self) -> Callable[[str, str, dict], None]:
        """Return a callback for the agent loop.

        Callback signature: ``callback(phase, detail, meta)``

        Phases:
            ``thinking``   – LLM is generating.
            ``tool_start`` – Tool execution starting.
            ``tool_end``   – Tool execution finished.
            ``done``       – Loop completed.
        """

        def _callback(phase: str, detail: str = "", meta: dict | None = None) -> None:
            meta = meta or {}
            if phase == "thinking":
                iteration = meta.get("iteration", "")
                if iteration and int(iteration) > 1:
                    self.update(f"nanobot is thinking... (step {iteration})")
                else:
                    self.update("nanobot is thinking...")
            elif phase == "tool_start":
                self.update(format_tool_status(detail, meta))
            elif phase == "tool_end":
                self.update("nanobot is thinking...")
            elif phase == "done":
                self.update("done")

        return _callback
