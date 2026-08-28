"""Shared state passed between the four agents."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

AgentName = Literal["orchestrator", "planner", "coder", "validator"]
ValidationStatus = Literal["not_started", "pending", "passed", "failed"]


@dataclass
class InputState:
    """What a caller sends in when starting a run."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class State(InputState):
    """Full workflow state, including retrieval evidence and validation progress."""

    current_agent: AgentName = "orchestrator"

    # Files retrieval has actually returned. The planner may only name files from
    # this list, which is how made-up paths get caught.
    retrieved_paths: list[str] = field(default_factory=list)

    plan: dict | None = None

    validation_status: ValidationStatus = "not_started"
    validation_attempts: int = 0
    max_validation_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_VALIDATION_ATTEMPTS", "2"))
    )
