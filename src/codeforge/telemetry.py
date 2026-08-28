"""Append-only JSONL run log used by the evaluation scripts."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def record(event: str, **values: Any) -> None:
    """Append one workflow event. Never writes secrets or source code."""
    output = Path(os.getenv("RUN_LOG", "run_log.jsonl"))
    entry = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **values}
    output.parent.mkdir(parents=True, exist_ok=True)
    with _lock, output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def summarize(log_path: str | Path) -> dict[str, Any]:
    """Roll a run log up into tool, retrieval, and validation counts."""
    events = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_type.setdefault(event["event"], []).append(event)

    tool_calls = by_type.get("tool_call", [])
    validations = by_type.get("validation", [])
    durations = [float(call.get("duration_ms", 0)) for call in tool_calls]
    return {
        "tool_calls": len(tool_calls),
        "tool_failures": sum(1 for call in tool_calls if not call.get("success")),
        "mean_tool_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "retrieval_queries": len(by_type.get("retrieval", [])),
        "rejected_plans": len(by_type.get("plan_rejected", [])),
        "validation_runs": len(validations),
        "validation_passes": sum(1 for run in validations if run.get("passed")),
    }
