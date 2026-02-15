"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SubagentInfo:
    """Rich metadata for a single subagent's lifecycle state."""

    # Identity
    task_id: str
    task: str
    label: str
    origin: dict[str, str]

    # Status
    status: str = "running"  # "running" | "completed" | "error"

    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 15

    # Current activity
    current_phase: str = "starting"  # "starting"|"thinking"|"tool_running"|"done"
    current_tool: str | None = None
    current_tool_args: dict | None = None

    # History
    tools_used: list[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None

    # Result
    result_summary: str | None = None
    error_message: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the subagent started."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def display_status(self) -> str:
        """Human-readable one-line status string."""
        if self.status != "running":
            return f"{self.status} ({self.elapsed_seconds:.1f}s)"
        if self.current_phase == "thinking":
            return f"thinking (step {self.iteration}/{self.max_iterations})"
        if self.current_phase == "tool_running" and self.current_tool:
            from nanobot.cli.activity import format_tool_status
            tool_display = format_tool_status(
                self.current_tool, self.current_tool_args or {}
            )
            return f"{tool_display} (step {self.iteration}/{self.max_iterations})"
        return f"{self.current_phase} (step {self.iteration}/{self.max_iterations})"


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SubagentTracker:
    """Centralized in-memory registry of subagent state.

    All methods are synchronous — safe to call from any async context.
    """

    def __init__(self, max_completed: int = 20) -> None:
        self._agents: dict[str, SubagentInfo] = {}
        self._max_completed = max_completed

    def register(self, info: SubagentInfo) -> None:
        """Register a new subagent at spawn time."""
        self._agents[info.task_id] = info

    def get(self, task_id: str) -> SubagentInfo | None:
        return self._agents.get(task_id)

    def update_phase(
        self,
        task_id: str,
        phase: str,
        *,
        iteration: int | None = None,
        tool_name: str | None = None,
        tool_args: dict | None = None,
    ) -> None:
        """Update the current phase of a subagent."""
        info = self._agents.get(task_id)
        if not info:
            return
        info.current_phase = phase
        if iteration is not None:
            info.iteration = iteration
        if phase == "tool_running":
            info.current_tool = tool_name
            info.current_tool_args = tool_args
            if tool_name:
                info.tools_used.append(tool_name)
        elif phase == "thinking":
            info.current_tool = None
            info.current_tool_args = None

    def mark_completed(
        self,
        task_id: str,
        status: str,
        result_summary: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark a subagent as completed or errored."""
        info = self._agents.get(task_id)
        if not info:
            return
        info.status = status
        info.current_phase = "done"
        info.current_tool = None
        info.current_tool_args = None
        info.ended_at = datetime.now()
        info.result_summary = result_summary
        info.error_message = error_message
        self._prune_completed()

    def get_running(self) -> list[SubagentInfo]:
        return [a for a in self._agents.values() if a.status == "running"]

    def get_all(self) -> list[SubagentInfo]:
        return list(self._agents.values())

    def get_running_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == "running")

    def _prune_completed(self) -> None:
        """Remove oldest completed entries when over the limit."""
        completed = [
            (tid, a) for tid, a in self._agents.items() if a.status != "running"
        ]
        if len(completed) > self._max_completed:
            completed.sort(key=lambda x: x[1].ended_at or x[1].started_at)
            for tid, _ in completed[: len(completed) - self._max_completed]:
                del self._agents[tid]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self.tracker = SubagentTracker()

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Register with tracker
        info = SubagentInfo(
            task_id=task_id,
            task=task,
            label=display_label,
            origin=origin,
        )
        self.tracker.register(info)

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(EditFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1
                self.tracker.update_phase(task_id, "thinking", iteration=iteration)

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Execute tools
                    for tool_call in response.tool_calls:
                        self.tracker.update_phase(
                            task_id, "tool_running",
                            iteration=iteration,
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                        )
                        args_str = json.dumps(tool_call.arguments)
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name} with arguments: {args_str}")
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            self.tracker.mark_completed(
                task_id, "completed", result_summary=final_result[:200],
            )
            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = str(e)
            self.tracker.mark_completed(
                task_id, "error", error_message=error_msg[:200],
            )
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, f"Error: {error_msg}", origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return self.tracker.get_running_count()
