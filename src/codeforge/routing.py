"""The two tools that move the workflow along.

One `handoff` tool with a target argument covers every destination, which keeps the
tool list short enough that small models still pick the right one.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

HANDOFF_TARGETS = ("orchestrator", "planner", "coder", "validator", "done")


@tool
async def handoff(
    target: str,
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Pass control to another role.

    Args:
        target: one of orchestrator, planner, coder, validator, done.
        reason: one line saying why control is moving.
    """
    if target not in HANDOFF_TARGETS:
        raise ValueError(f"target must be one of {', '.join(HANDOFF_TARGETS)}")
    destination = "__end__" if target == "done" else target
    return Command(
        goto=destination,
        update={
            "messages": [
                ToolMessage(content=f"Handing off to {target}: {reason}", tool_call_id=tool_call_id)
            ],
            "current_agent": target if target != "done" else "orchestrator",
        },
    )


@tool
async def submit_plan(
    summary: str,
    files_to_touch: list[str],
    steps: list[str],
    test_plan: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Submit the implementation plan and send it to the coder.

    Args:
        summary: what the change does, in one or two sentences.
        files_to_touch: repository relative paths, each one already seen in retrieval.
        steps: ordered implementation steps.
        test_plan: how the change will be proved to work.
    """
    plan = {
        "summary": summary,
        "files_to_touch": files_to_touch,
        "steps": steps,
        "test_plan": test_plan,
    }
    return Command(
        goto="coder",
        update={
            "plan": plan,
            "current_agent": "coder",
            "messages": [
                ToolMessage(content=f"Plan accepted: {summary}", tool_call_id=tool_call_id)
            ],
        },
    )


WORKFLOW_TOOLS: list[BaseTool] = [handoff, submit_plan]
