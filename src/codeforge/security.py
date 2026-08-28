"""Path and remote-write permission checks applied before any MCP tool call."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROTECTED_BRANCHES = {"main", "master", "production", "release"}
REMOTE_WRITE_TOOLS = {
    "add_issue_comment",
    "create_branch",
    "create_issue",
    "create_or_update_file",
    "create_pull_request",
    "create_pull_request_review",
    "create_repository",
    "fork_repository",
    "merge_pull_request",
    "push_files",
    "update_issue",
    "update_pull_request_branch",
}
PATH_ARGUMENTS = {"directory", "file_path", "path", "repo_path", "repository_path", "root"}
BRANCH_ARGUMENTS = ("branch", "branch_name", "target_branch")


def repository_root() -> Path:
    """Return the one directory the agent is allowed to touch."""
    configured = os.getenv("REPOSITORY_ROOT")
    if not configured:
        raise RuntimeError("REPOSITORY_ROOT must be configured")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"REPOSITORY_ROOT is not a directory: {root}")
    return root


def resolve_repository_path(path: str | Path) -> Path:
    """Resolve a path and reject anything that escapes the repository root."""
    root = repository_root()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"Path is outside REPOSITORY_ROOT: {candidate}")
    return candidate


def is_remote_write_tool(tool_name: str) -> bool:
    """Return whether a tool writes to a remote GitHub repository."""
    normalized = tool_name.lower().replace("-", "_")
    return normalized in REMOTE_WRITE_TOOLS or normalized.startswith("github_write_")


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    """Raise if a call leaves the repository root or targets a protected branch."""
    for key, value in arguments.items():
        if key in PATH_ARGUMENTS and isinstance(value, str):
            resolve_repository_path(value)

    if not is_remote_write_tool(tool_name):
        return

    for key in BRANCH_ARGUMENTS:
        value = arguments.get(key)
        if isinstance(value, str) and value.lower() in PROTECTED_BRANCHES:
            raise PermissionError(f"Direct writes to protected branch '{value}' are blocked")
