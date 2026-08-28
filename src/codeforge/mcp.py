"""Start the configured MCP servers and expose their tools to the agents.

Tools come from the official multi-server client in langchain-mcp-adapters, which
launches each stdio server and converts its schemas into LangChain tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "mcp_servers.json"


def load_server_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the server list and expand ${VARIABLE} placeholders from the environment."""
    config_path = Path(path or os.getenv("MCP_SERVERS_FILE", DEFAULT_CONFIG))
    servers = json.loads(config_path.read_text(encoding="utf-8"))["servers"]

    connections = {}
    for name, spec in servers.items():
        overrides = {key: os.path.expandvars(value) for key, value in spec.get("env", {}).items()}
        connections[name] = {
            "transport": "stdio",
            "command": os.path.expandvars(spec["command"]),
            "args": [os.path.expandvars(argument) for argument in spec.get("args", [])],
            "env": {**os.environ, **overrides} if overrides else None,
        }
    return connections


async def load_tools(path: str | Path | None = None) -> list[BaseTool]:
    """Connect to every configured MCP server and return its tools."""
    client = MultiServerMCPClient(load_server_config(path))  # type: ignore[arg-type]
    return await client.get_tools()
