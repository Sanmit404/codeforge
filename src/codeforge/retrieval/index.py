"""Hybrid repository index: dense vectors from Chroma fused with BM25 keyword hits."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

import chromadb

from codeforge.retrieval.chunker import CodeChunk, chunk_python_file
from codeforge.retrieval.lexical import BM25, reciprocal_rank_fusion
from codeforge.retrieval.scanner import scan_python_files


class Embedder(Protocol):
    """Minimal embedding interface, so tests can pass a fake."""

    def __call__(self, input: list[str]) -> Any: ...


class RepositoryIndex:
    """Index Python code and retrieve it with file, symbol, and line evidence."""

    def __init__(self, db_path: str | Path | None = None, embedder: Embedder | None = None) -> None:
        self.db_path = Path(db_path if db_path is not None else os.getenv("INDEX_PATH", ".codeforge/chroma"))
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._corpus_cache: dict[str, tuple[list[str], list[str], list[dict[str, Any]], BM25]] = {}
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            "repository_code", metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> Embedder:
        """Load the embedding model only when it is first needed."""
        if self._embedder is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._embedder = DefaultEmbeddingFunction()
        return self._embedder

    def _embed(self, values: list[str]) -> list[list[float]]:
        vectors = self.embedder(input=values)
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    @staticmethod
    def _repository_id(repository: Path) -> str:
        return hashlib.sha256(str(repository).encode()).hexdigest()[:16]

    @staticmethod
    def _commit(repository: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "working-tree"

    @staticmethod
    def _document(chunk: CodeChunk) -> str:
        return f"Path: {chunk.path}\nSymbol: {chunk.symbol}\n{chunk.content}"

    def index(self, repository_path: str | Path) -> dict[str, Any]:
        """Rebuild the index for one allowed repository."""
        started = time.perf_counter()
        repository, files = scan_python_files(repository_path)
        repository_id = self._repository_id(repository)
        commit = self._commit(repository)
        chunks = [chunk for path in files for chunk in chunk_python_file(path, repository)]

        self.collection.delete(where={"repository_id": repository_id})
        self._corpus_cache.pop(repository_id, None)
        summary = {
            "repository": str(repository),
            "files": len(files),
            "chunks": len(chunks),
            "commit": commit,
            "duration_ms": 0.0,
        }
        if chunks:
            documents = [self._document(chunk) for chunk in chunks]
            self.collection.upsert(
                ids=[
                    hashlib.sha256(
                        f"{repository_id}:{chunk.path}:{chunk.symbol}:{chunk.start_line}".encode()
                    ).hexdigest()
                    for chunk in chunks
                ],
                documents=documents,
                embeddings=self._embed(documents),  # type: ignore[arg-type]
                metadatas=[  # type: ignore[arg-type]
                    {
                        "repository_id": repository_id,
                        "path": chunk.path,
                        "symbol": chunk.symbol,
                        "kind": chunk.kind,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "commit": commit,
                    }
                    for chunk in chunks
                ],
            )
        summary["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return summary

    def _corpus(self, repository_id: str) -> tuple[list[str], list[str], list[dict[str, Any]], BM25]:
        """Load this repository's chunks once and keep the BM25 model around."""
        if repository_id not in self._corpus_cache:
            raw = self.collection.get(
                where={"repository_id": repository_id}, include=["documents", "metadatas"]
            )
            ids = list(raw.get("ids") or [])
            documents = list(raw.get("documents") or [])
            metadatas = [dict(item) for item in (raw.get("metadatas") or [])]
            self._corpus_cache[repository_id] = (ids, documents, metadatas, BM25(documents))
        return self._corpus_cache[repository_id]

    def _dense_ranking(self, query: str, repository_id: str, pool: int, ids: list[str]) -> list[int]:
        position = {document_id: index for index, document_id in enumerate(ids)}
        result = self.collection.query(
            query_embeddings=self._embed([query]),  # type: ignore[arg-type]
            n_results=min(pool, max(len(ids), 1)),
            where={"repository_id": repository_id},
            include=[],
        )
        found = (result["ids"] or [[]])[0]
        return [position[document_id] for document_id in found if document_id in position]

    def search(
        self,
        repository_path: str | Path,
        query: str,
        top_k: int = 8,
        path_prefix: str = "",
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Retrieve code for a question. Mode is dense, lexical, or hybrid."""
        started = time.perf_counter()
        repository, _ = scan_python_files(repository_path)
        repository_id = self._repository_id(repository)
        top_k = max(1, min(top_k, 12))
        ids, documents, metadatas, bm25 = self._corpus(repository_id)
        if not documents:
            return {"query": query, "mode": mode, "matches": [], "duration_ms": 0.0}

        pool = max(top_k * 5, 25)
        dense = self._dense_ranking(query, repository_id, pool, ids) if mode != "lexical" else []
        lexical: list[int] = []
        if mode != "dense":
            scores = bm25.scores(query)
            ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            lexical = [index for index in ordered[:pool] if scores[index] > 0]

        if mode == "dense":
            order = dense
        elif mode == "lexical":
            order = lexical
        else:
            order = reciprocal_rank_fusion([dense, lexical])

        matches = []
        for rank, index in enumerate(order, start=1):
            metadata = metadatas[index]
            if path_prefix and not str(metadata["path"]).startswith(path_prefix):
                continue
            found_in = [
                name for name, ranking in (("dense", dense), ("lexical", lexical)) if index in ranking
            ]
            matches.append({**metadata, "rank": rank, "found_by": found_in, "content": documents[index]})
            if len(matches) == top_k:
                break

        return {
            "query": query,
            "mode": mode,
            "matches": matches,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
