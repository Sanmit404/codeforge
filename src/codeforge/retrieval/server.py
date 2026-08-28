"""MCP tools for indexing and searching the configured repository."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeforge.retrieval.index import RepositoryIndex

mcp = FastMCP("repository-index")
index = RepositoryIndex()


@mcp.tool()
def index_repository(repository_path: str) -> dict:
    """Index every Python module, class, function, and method in an allowed repository."""
    return index.index(repository_path)


@mcp.tool()
def search_repository(
    repository_path: str,
    query: str,
    top_k: int = 8,
    path_prefix: str = "",
    mode: str = "hybrid",
) -> dict:
    """Retrieve code with path, symbol, and line evidence. Mode is dense, lexical, or hybrid."""
    return index.search(repository_path, query, top_k, path_prefix, mode)


@mcp.tool()
def refresh_repository(repository_path: str) -> dict:
    """Rebuild the index after the coder changes files."""
    return index.index(repository_path)


if __name__ == "__main__":
    mcp.run()
