"""MCP server exposing the one validation tool."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeforge.validation.runner import validate_repository as run_validation

mcp = FastMCP("validation")


@mcp.tool()
def validate_repository(repository_path: str) -> dict:
    """Run pytest and Ruff against the repository and report what failed."""
    return run_validation(repository_path)


if __name__ == "__main__":
    mcp.run()
