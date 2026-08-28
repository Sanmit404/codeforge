import numpy as np
import pytest

from codeforge.retrieval.index import RepositoryIndex


class TopicEmbedder:
    """Stands in for the real model so tests stay fast and offline."""

    def __call__(self, input):
        vectors = []
        for value in input:
            lowered = value.lower()
            vectors.append(
                [
                    float("auth" in lowered or "login" in lowered),
                    float("payment" in lowered or "charge" in lowered),
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype=float)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "auth.py").write_text(
        "def login(user):\n    return user\n\n\ndef refresh_token(session):\n    return session\n"
    )
    (root / "payments.py").write_text("def charge_payment(amount):\n    return amount\n")
    monkeypatch.setenv("REPOSITORY_ROOT", str(root))
    return root


@pytest.fixture
def index(tmp_path, repository):
    built = RepositoryIndex(db_path=tmp_path / "chroma", embedder=TopicEmbedder())
    built.index(repository)
    return built


def test_index_reports_files_and_chunks(index, repository):
    summary = index.index(repository)

    assert summary["files"] == 2
    # login, refresh_token, charge_payment. Neither file has module level code,
    # so neither gets a module chunk.
    assert summary["chunks"] == 3


def test_search_returns_citable_evidence(index, repository):
    match = index.search(repository, "where does login happen", top_k=3)["matches"][0]

    assert match["path"] == "auth.py"
    assert match["start_line"] >= 1
    assert match["found_by"]


def test_keyword_pass_finds_an_identifier_the_embedder_misses(index, repository):
    dense = index.search(repository, "refresh_token", top_k=3, mode="dense")["matches"]
    lexical = index.search(repository, "refresh_token", top_k=3, mode="lexical")["matches"]

    assert lexical[0]["symbol"] == "refresh_token"
    assert dense[0]["symbol"] != "refresh_token"


def test_hybrid_recovers_the_identifier_match(index, repository):
    matches = index.search(repository, "refresh_token", top_k=3, mode="hybrid")["matches"]

    assert "refresh_token" in {match["symbol"] for match in matches}


def test_path_prefix_limits_results(index, repository):
    matches = index.search(repository, "charge", top_k=5, path_prefix="payments")["matches"]

    assert matches
    assert all(match["path"].startswith("payments") for match in matches)
