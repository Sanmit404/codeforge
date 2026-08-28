"""Fixed validation commands. The agent cannot ask this server to run anything else."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from codeforge.security import resolve_repository_path

ALLOWED_EXECUTABLES = {sys.executable, "ruff"}
FAILING_TEST = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def _run(command: list[str], repository: Path, timeout: int) -> dict[str, Any]:
    """Run one allowlisted command and summarize the result."""
    started = time.perf_counter()
    executable = command[0]
    if executable not in ALLOWED_EXECUTABLES:
        raise PermissionError(f"Validation command is not allowlisted: {executable}")
    if executable == "ruff" and shutil.which(executable) is None:
        return {"command": " ".join(command), "status": "skipped", "output": "ruff is not installed"}

    try:
        result = subprocess.run(
            command, cwd=repository, capture_output=True, text=True, timeout=timeout, check=False
        )
        status = "passed" if result.returncode == 0 else "failed"
        output = f"{result.stdout}\n{result.stderr}".strip()
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}".strip()

    return {
        "command": " ".join(command),
        "status": status,
        "output": output[-6000:],
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def changed_python_files(repository: Path) -> list[str]:
    """Python files the coder has touched, so linting stays focused on new work."""
    result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    paths = []
    for line in result.stdout.splitlines():
        path = line[3:].split(" -> ")[-1].strip()
        if path.endswith(".py") and (repository / path).exists():
            paths.append(path)
    return paths


def validate_repository(repository_path: str) -> dict[str, Any]:
    """Run the test suite and the linter, and name the tests that failed."""
    repository = resolve_repository_path(repository_path)
    timeout = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "120"))
    changed = changed_python_files(repository)

    checks = [
        _run([sys.executable, "-m", "pytest", "-q"], repository, timeout),
        _run(["ruff", "check", *(changed or ["."])], repository, timeout),
    ]
    failed = [check for check in checks if check["status"] in {"failed", "timeout"}]
    return {
        "repository": str(repository),
        "passed": not failed,
        "changed_files": changed,
        "failing_tests": sorted(set(FAILING_TEST.findall(checks[0]["output"]))),
        "checks": checks,
    }
