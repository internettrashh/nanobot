"""Status tool for querying subagent state."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentTracker


class SubagentStatusTool(Tool):
    """
    Tool to query the status of running and recently-completed subagents.

    The main agent can call this to check what background tasks are doing
    without waiting for them to complete.
    """

    def __init__(self, tracker: "SubagentTracker") -> None:
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "subagent_status"

    @property
    def description(self) -> str:
        return (
            "Check the status of background subagents. "
            "Actions: 'list' shows running subagents, "
            "'detail' shows full info for a specific subagent by ID, "
            "'all' shows running plus recently completed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "detail", "all"],
                    "description": (
                        "Action to perform: 'list' for running agents, "
                        "'detail' for one agent, 'all' for running + completed"
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": "Subagent ID (required for 'detail' action)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._list_running()
        elif action == "detail":
            return self._detail(task_id)
        elif action == "all":
            return self._list_all()
        return f"Unknown action: {action}"

    def _list_running(self) -> str:
        agents = self._tracker.get_running()
        if not agents:
            return "No subagents currently running."
        lines = []
        for a in agents:
            elapsed = f"{a.elapsed_seconds:.0f}s"
            lines.append(
                f"- [{a.task_id}] {a.label} | {a.display_status} | {elapsed} elapsed"
            )
        return f"{len(agents)} running subagent(s):\n" + "\n".join(lines)

    def _detail(self, task_id: str | None) -> str:
        if not task_id:
            return "Error: task_id is required for 'detail' action."
        info = self._tracker.get(task_id)
        if not info:
            return f"No subagent found with ID '{task_id}'."

        task_preview = info.task[:100] + ("..." if len(info.task) > 100 else "")
        lines = [
            f"Subagent [{info.task_id}]: {info.label}",
            f"  Status: {info.display_status}",
            f"  Task: {task_preview}",
            f"  Iteration: {info.iteration}/{info.max_iterations}",
            f"  Elapsed: {info.elapsed_seconds:.1f}s",
            f"  Tools used: {', '.join(info.tools_used) if info.tools_used else 'none'}",
        ]
        if info.current_tool:
            from nanobot.cli.activity import format_tool_status
            lines.append(
                f"  Currently: {format_tool_status(info.current_tool, info.current_tool_args or {})}"
            )
        if info.result_summary:
            lines.append(f"  Result: {info.result_summary}")
        if info.error_message:
            lines.append(f"  Error: {info.error_message}")
        return "\n".join(lines)

    def _list_all(self) -> str:
        agents = self._tracker.get_all()
        if not agents:
            return "No subagents tracked."
        running = [a for a in agents if a.status == "running"]
        completed = [a for a in agents if a.status != "running"]

        lines: list[str] = []
        if running:
            lines.append(f"Running ({len(running)}):")
            for a in running:
                lines.append(
                    f"  - [{a.task_id}] {a.label} | {a.display_status}"
                )
        if completed:
            lines.append(f"Completed ({len(completed)}):")
            for a in completed:
                elapsed = f"{a.elapsed_seconds:.0f}s"
                status_icon = "ok" if a.status == "completed" else "ERR"
                lines.append(
                    f"  - [{a.task_id}] {a.label} | {status_icon} | {elapsed}"
                )
        return "\n".join(lines)
