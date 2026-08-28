"""Repository scanning for the Python-only index."""

from __future__ import annotations

from pathlib import Path

from codeforge.security import resolve_repository_path

IGNORED_DIRECTORIES = {
    ".codeforge",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def scan_python_files(repository_path: str | Path) -> tuple[Path, list[Path]]:
    """Return allowed Python files under a repository in a stable order."""
    repository = resolve_repository_path(repository_path)
    files = [
        path
        for path in repository.rglob("*.py")
        if path.is_file()
        and not any(part in IGNORED_DIRECTORIES for part in path.relative_to(repository).parts)
    ]
    return repository, sorted(files)
