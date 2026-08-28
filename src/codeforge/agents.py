"""Build the four agent nodes from one prompt registry.

Each role is the same model bound to a different system prompt, so adding a role
means adding a prompt.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from codeforge.prompts import PROMPTS
from codeforge.state import State

AgentNode = Callable[[State], Awaitable[dict[str, Any]]]


def load_chat_model(model_name: str | None = None) -> BaseChatModel:
    """Build a chat model from a "provider/model" string."""
    name = model_name or os.getenv("LLM_MODEL", "openrouter/openai/gpt-4o-mini")
    provider, _, model = name.partition("/")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=model)  # type: ignore[call-arg]
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model)
    if provider in {"openai", "openrouter"}:
        from langchain_openai import ChatOpenAI

        if provider == "openai":
            return ChatOpenAI(model=model)
        return ChatOpenAI(
            model=model,
            api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore[arg-type]
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    raise ValueError(f"Unsupported model provider: {provider}")


def _status_note(role: str, state: State) -> SystemMessage:
    plan = state.plan["summary"] if state.plan else "none yet"
    return SystemMessage(
        content=(
            f"You are acting as the {role}. "
            f"Validation: {state.validation_status}, repair attempts "
            f"{state.validation_attempts}/{state.max_validation_attempts}. "
            f"Accepted plan: {plan}. "
            f"Files seen in retrieval so far: {', '.join(state.retrieved_paths) or 'none'}."
        )
    )


def build_agent(role: str, llm: BaseChatModel, tools: list[BaseTool]) -> AgentNode:
    """Return the graph node for one role."""
    model = llm.bind_tools(tools)
    system = SystemMessage(content=PROMPTS[role])

    async def run(state: State) -> dict[str, Any]:
        response = await model.ainvoke([system, *state.messages, _status_note(role, state)])
        return {"messages": [response], "current_agent": role}

    return run
