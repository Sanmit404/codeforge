import pytest

from codeforge.validation.runner import validate_repository


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    monkeypatch.setenv("REPOSITORY_ROOT", str(root))
    monkeypatch.setenv("VALIDATION_TIMEOUT_SECONDS", "60")
    return root


def test_a_working_repository_passes(repository):
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )

    result = validate_repository(str(repository))

    assert result["passed"] is True
    assert result["failing_tests"] == []


def test_a_broken_test_is_reported_by_name(repository):
    (repository / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 6\n"
    )

    result = validate_repository(str(repository))

    assert result["passed"] is False
    assert any("test_add" in name for name in result["failing_tests"])


def test_repositories_outside_the_root_are_refused(repository, tmp_path):
    with pytest.raises(PermissionError):
        validate_repository(str(tmp_path / "elsewhere"))
