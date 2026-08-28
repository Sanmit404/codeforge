"""LangGraph workflow that connects the four agents to the MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from codeforge import telemetry
from codeforge.agents import build_agent, load_chat_model
from codeforge.mcp import load_tools
from codeforge.routing import WORKFLOW_TOOLS
from codeforge.security import is_remote_write_tool, validate_tool_arguments
from codeforge.state import State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLES = ("orchestrator", "planner", "coder", "validator")
WORKFLOW_TOOL_NAMES = {tool.name for tool in WORKFLOW_TOOLS}


def _as_json(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return None


def _ask_human(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Pause the run until a person approves a remote write."""
    preview = {
        key: (value[:1000] if isinstance(value, str) else value)
        for key, value in arguments.items()
        if "token" not in key.lower()
    }
    decision = interrupt(
        {
            "type": "remote_write_approval",
            "tool": tool_name,
            "arguments": preview,
            "question": "Approve this remote GitHub write?",
        }
    )
    return decision.get("approved") is True if isinstance(decision, dict) else decision is True


class Workflow:
    """Owns the agent nodes and the single tool execution node."""

    def __init__(self, tools: list[BaseTool]) -> None:
        llm = load_chat_model()
        self.tools = {tool.name: tool for tool in tools}
        self.agents = {role: build_agent(role, llm, tools) for role in ROLES}

    async def run_tools(self, state: State) -> Command:
        """Run the first requested tool. Extra calls in the same turn are declined."""
        calls = list(getattr(state.messages[-1], "tool_calls", None) or [])
        if not calls:
            return Command(goto=state.current_agent)

        call = calls[0]
        try:
            command = await self._dispatch(state, call["name"], call.get("args") or {}, call["id"])
        except Exception as exc:
            telemetry.record("tool_call", tool=call["name"], success=False, error=str(exc))
            logger.exception("Tool %s failed", call["name"])
            command = Command(
                goto=state.current_agent,
                update={
                    "messages": [
                        ToolMessage(content=f"Error: {exc}", tool_call_id=call["id"], name=call["name"])
                    ]
                },
            )

        update = dict(command.update or {})
        update["messages"] = list(update.get("messages", [])) + [
            ToolMessage(
                content="Skipped. This workflow runs one tool per turn, ask again next turn.",
                tool_call_id=extra["id"],
                name=extra["name"],
            )
            for extra in calls[1:]
        ]
        return Command(goto=command.goto, update=update)

    async def _dispatch(
        self, state: State, name: str, arguments: dict[str, Any], call_id: str
    ) -> Command:
        validate_tool_arguments(name, arguments)

        if is_remote_write_tool(name):
            if state.validation_status != "passed":
                raise PermissionError("Remote writes require a passed validation run")
            if not _ask_human(name, arguments):
                raise PermissionError("Remote write rejected by the human operator")

        if name == "submit_plan":
            rejection = self._reject_ungrounded_plan(state, arguments, call_id)
            if rejection is not None:
                return rejection

        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        payload: Any = arguments
        if name in WORKFLOW_TOOL_NAMES:
            payload = {"args": arguments, "name": name, "type": "tool_call", "id": call_id}

        started = time.perf_counter()
        result = await tool.ainvoke(payload)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        telemetry.record("tool_call", tool=name, duration_ms=duration_ms, success=True)

        if isinstance(result, Command):
            return result

        content = result if isinstance(result, str) else json.dumps(result, default=str)
        update: dict[str, Any] = {
            "messages": [ToolMessage(content=content, tool_call_id=call_id, name=name)],
            "current_agent": state.current_agent,
        }
        goto = state.current_agent
        body = _as_json(content)

        if name == "search_repository" and isinstance(body, dict):
            matches = body.get("matches", [])
            found = {match["path"] for match in matches if "path" in match}
            update["retrieved_paths"] = sorted(set(state.retrieved_paths) | found)
            telemetry.record("retrieval", matches=len(matches), mode=body.get("mode"))

        if name == "validate_repository" and isinstance(body, dict):
            goto = self._apply_validation(state, body, update)

        return Command(goto=goto, update=update)

    @staticmethod
    def _reject_ungrounded_plan(
        state: State, arguments: dict[str, Any], call_id: str
    ) -> Command | None:
        """Refuse a plan that names files retrieval never returned."""
        unknown = [
            path
            for path in arguments.get("files_to_touch") or []
            if path not in state.retrieved_paths
        ]
        if not unknown:
            return None
        telemetry.record("plan_rejected", unknown=unknown)
        return Command(
            goto="planner",
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Plan rejected. Retrieval never returned these paths: "
                            f"{', '.join(unknown)}. Search for them first, then submit again."
                        ),
                        tool_call_id=call_id,
                        name="submit_plan",
                    )
                ],
                "current_agent": "planner",
            },
        )

    @staticmethod
    def _apply_validation(state: State, body: dict[str, Any], update: dict[str, Any]) -> str:
        """Set the pass or fail status from the tool output."""
        passed = bool(body.get("passed"))
        telemetry.record("validation", passed=passed, attempt=state.validation_attempts)
        if passed:
            update["validation_status"] = "passed"
            return state.current_agent

        attempts = state.validation_attempts + 1
        update["validation_status"] = "failed"
        update["validation_attempts"] = attempts
        if attempts < state.max_validation_attempts:
            return state.current_agent

        update["messages"].append(
            HumanMessage(content="Repair limit reached, a human needs to review this run.")
        )
        update["current_agent"] = "orchestrator"
        return "orchestrator"


def _needs_tools(state: State) -> str:
    message = state.messages[-1]
    return "tools" if isinstance(message, AIMessage) and message.tool_calls else "__end__"


def build_graph(tools: list[BaseTool]):
    """Wire the four roles to the shared tool node."""
    workflow = Workflow(tools)
    builder = StateGraph(State)
    for role in ROLES:
        builder.add_node(role, workflow.agents[role])
        builder.add_conditional_edges(role, _needs_tools, {"tools": "tools", "__end__": END})
    builder.add_node("tools", workflow.run_tools)
    builder.add_edge("__start__", "orchestrator")
    return builder.compile()


async def make_graph():
    """Entry point named in langgraph.json. Starting the MCP servers is the slow part."""
    load_dotenv()
    return build_graph(await load_tools() + WORKFLOW_TOOLS)


async def run_once(task: str) -> None:
    """Run one request from the command line instead of LangGraph Studio."""
    compiled = await make_graph()
    initial = State(messages=[HumanMessage(content=task)])
    async for update in compiled.astream(initial, stream_mode="updates", config={"recursion_limit": 60}):
        logger.info("update: %s", update)


if __name__ == "__main__":
    asyncio.run(run_once(" ".join(sys.argv[1:])))
