"""Split Python files into retrievable chunks using the AST."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

FunctionNode = (ast.FunctionDef, ast.AsyncFunctionDef)
TOP_LEVEL_CONTEXT = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)


@dataclass(frozen=True)
class CodeChunk:
    """One retrievable unit of source with enough metadata to cite it."""

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str


def _segment(lines: list[str], node: ast.AST) -> str:
    start = max(getattr(node, "lineno", 1) - 1, 0)
    return "".join(lines[start : getattr(node, "end_lineno", start + 1)]).strip()


def _class_header(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    docstring = ast.get_docstring(node, clean=False)
    return f'{header}\n    """{docstring}"""' if docstring else header


def _collect(body: list[ast.stmt], lines: list[str], path: str, prefix: str) -> list[CodeChunk]:
    """Walk class bodies so nested classes and methods get qualified names."""
    chunks: list[CodeChunk] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            symbol = f"{prefix}{node.name}"
            chunks.append(
                CodeChunk(path, symbol, "class", node.lineno, node.lineno, _class_header(node))
            )
            chunks.extend(_collect(node.body, lines, path, f"{symbol}."))
        elif isinstance(node, FunctionNode):
            chunks.append(
                CodeChunk(
                    path,
                    f"{prefix}{node.name}",
                    "method" if prefix else "function",
                    node.lineno,
                    node.end_lineno or node.lineno,
                    _segment(lines, node),
                )
            )
    return chunks


def chunk_python_file(path: Path, repository: Path) -> list[CodeChunk]:
    """Split one file into a module chunk plus one chunk per class and function."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    relative_path = path.relative_to(repository).as_posix()
    lines = text.splitlines(keepends=True)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [CodeChunk(relative_path, "<module>", "module", 1, len(lines), text)]

    chunks = _collect(tree.body, lines, relative_path, "")
    header = [ast.get_docstring(tree, clean=False) or ""]
    header += [_segment(lines, node) for node in tree.body if isinstance(node, TOP_LEVEL_CONTEXT)]
    module_content = "\n".join(part for part in header if part).strip()
    if module_content or not chunks:
        chunks.insert(
            0,
            CodeChunk(relative_path, "<module>", "module", 1, len(lines), module_content),
        )
    return chunks
