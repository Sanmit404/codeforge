import subprocess

from codeforge.retrieval.benchmark import evaluate, load_cases


class StubIndex:
    """Returns a fixed ranking so the metric maths can be checked on its own."""

    def __init__(self, paths):
        self.paths = paths

    def search(self, repository, query, top_k, mode):
        return {"matches": [{"path": path} for path in self.paths[:top_k]]}


def _git(repository, *arguments):
    subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)


def test_commits_become_labelled_retrieval_cases(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "dev@example.com")
    _git(repository, "config", "user.name", "dev")
    (repository / "auth.py").write_text("def login():\n    pass\n")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-qm", "add login endpoint")

    cases = load_cases(repository)

    assert cases == [{"query": "add login endpoint", "files": ["auth.py"]}]


def test_commits_touching_too_many_files_are_skipped(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "dev@example.com")
    _git(repository, "config", "user.name", "dev")
    for name in ["a.py", "b.py", "c.py"]:
        (repository / name).write_text("x = 1\n")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-qm", "big refactor")

    assert load_cases(repository, max_files=2) == []


def test_metrics_reward_ranking_the_right_file_first(tmp_path):
    cases = [{"query": "add login endpoint", "files": ["auth.py"]}]

    good = evaluate(StubIndex(["auth.py", "other.py"]), tmp_path, cases, modes=("hybrid",))
    poor = evaluate(StubIndex(["other.py", "auth.py"]), tmp_path, cases, modes=("hybrid",))

    assert good["modes"]["hybrid"] == {"recall_at_k": 1.0, "mrr": 1.0}
    assert poor["modes"]["hybrid"] == {"recall_at_k": 1.0, "mrr": 0.5}
