from pathlib import Path

import pytest

from codeforge.security import (
    is_remote_write_tool,
    resolve_repository_path,
    validate_tool_arguments,
)


def test_paths_stay_inside_the_repository_root(monkeypatch, tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setenv("REPOSITORY_ROOT", str(repository))

    assert resolve_repository_path("src/file.py") == repository / "src" / "file.py"
    with pytest.raises(PermissionError):
        resolve_repository_path(tmp_path / "outside.py")


def test_missing_root_is_an_error(monkeypatch):
    monkeypatch.delenv("REPOSITORY_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        resolve_repository_path("src/file.py")


def test_protected_branch_writes_are_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path))
    with pytest.raises(PermissionError):
        validate_tool_arguments("push_files", {"branch": "main", "path": "src/app.py"})


def test_feature_branch_writes_are_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOSITORY_ROOT", str(tmp_path))
    validate_tool_arguments(
        "push_files", {"branch": "feature/auth", "path": Path("src/app.py").as_posix()}
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("create_pull_request", True),
        ("github_write_anything", True),
        ("create-branch", True),
        ("search_repository", False),
        ("validate_repository", False),
    ],
)
def test_remote_write_detection(name, expected):
    assert is_remote_write_tool(name) is expected
