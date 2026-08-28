"""Retrieval evaluation with ground truth mined from the repository's own git history.

A commit gives us a free labelled example: the commit subject reads like a feature
request, and the Python files that commit touched are the files a good retriever
should surface. That means any git repository can be used as a test set without
anyone hand labelling queries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from codeforge.retrieval.index import RepositoryIndex

COMMIT_MARKER = "__commit__"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout


def load_cases(repository: Path, limit: int = 60, max_files: int = 5) -> list[dict[str, Any]]:
    """Turn recent commits into query and relevant-files pairs."""
    output = _git(
        repository,
        "log",
        f"-{limit}",
        "--no-merges",
        f"--pretty=format:{COMMIT_MARKER}%x1f%s",
        "--name-only",
    )
    cases: list[dict[str, Any]] = []
    subject = ""
    files: list[str] = []

    def flush() -> None:
        existing = [path for path in files if (repository / path).exists()]
        if subject and 1 <= len(existing) <= max_files:
            cases.append({"query": subject, "files": sorted(set(existing))})

    for line in output.splitlines():
        if line.startswith(COMMIT_MARKER):
            flush()
            subject = line.split("\x1f", 1)[-1].strip()
            files = []
        elif line.strip().endswith(".py"):
            files.append(line.strip())
    flush()
    return cases


def evaluate(
    index: RepositoryIndex,
    repository: Path,
    cases: list[dict[str, Any]],
    top_k: int = 10,
    modes: tuple[str, ...] = ("dense", "lexical", "hybrid"),
) -> dict[str, Any]:
    """Report recall@k and mean reciprocal rank for each retrieval mode."""
    report: dict[str, Any] = {"cases": len(cases), "top_k": top_k, "modes": {}}
    for mode in modes:
        recalls, reciprocals = [], []
        for case in cases:
            matches = index.search(repository, case["query"], top_k=top_k, mode=mode)["matches"]
            paths = [match["path"] for match in matches]
            gold = set(case["files"])
            recalls.append(len(gold.intersection(paths)) / len(gold))
            rank = next((i for i, path in enumerate(paths, start=1) if path in gold), 0)
            reciprocals.append(1 / rank if rank else 0.0)
        count = len(cases) or 1
        report["modes"][mode] = {
            "recall_at_k": round(sum(recalls) / count, 4),
            "mrr": round(sum(reciprocals) / count, 4),
        }
    return report


def main() -> None:
    """Index a repository, mine its history, and print the comparison table."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m codeforge.retrieval.benchmark <repository> [limit]")
    repository = Path(sys.argv[1]).resolve()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    index = RepositoryIndex()
    index.index(repository)
    cases = load_cases(repository, limit=limit)
    if not cases:
        raise SystemExit("No usable commits found in history")
    print(json.dumps(evaluate(index, repository, cases), indent=2))


if __name__ == "__main__":
    main()
