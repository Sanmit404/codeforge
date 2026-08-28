from codeforge.retrieval.chunker import chunk_python_file

SOURCE = '''"""Authentication service."""
import os


def issue_token(user_id):
    return user_id


class AuthService:
    """Checks tokens."""

    def validate(self, token):
        return bool(token)

    class Cache:
        def get(self, key):
            return key
'''


def test_chunks_carry_symbols_and_line_numbers(tmp_path):
    source = tmp_path / "service.py"
    source.write_text(SOURCE)

    chunks = {chunk.symbol: chunk for chunk in chunk_python_file(source, tmp_path)}

    assert chunks["issue_token"].kind == "function"
    assert chunks["issue_token"].start_line == 5
    assert chunks["<module>"].content.startswith("Authentication service.")
    assert "import os" in chunks["<module>"].content


def test_nested_definitions_get_qualified_names(tmp_path):
    source = tmp_path / "service.py"
    source.write_text(SOURCE)

    symbols = {chunk.symbol for chunk in chunk_python_file(source, tmp_path)}

    assert "AuthService.validate" in symbols
    assert "AuthService.Cache" in symbols
    assert "AuthService.Cache.get" in symbols


def test_unparsable_file_still_produces_one_chunk(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n")

    chunks = chunk_python_file(source, tmp_path)

    assert len(chunks) == 1
    assert chunks[0].symbol == "<module>"
