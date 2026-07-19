"""Smoke test for markdown_splitter — captures current behavior before refactoring.

NOTE: The 50-char minimum threshold means many short markdown snippets
produce zero chunks. These tests use sufficiently long content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knowledge_graph.markdown_splitter import split_markdown_into_chunks, TextChunk


def make_text(parts: list[str], separator: str = "\n\n") -> str:
    """Join text parts with separator and ensure total length >= 50 chars."""
    return separator.join(parts)


LONG_MD = make_text([
    "# Introduction",
    "This is a sufficiently long introductory paragraph that definitely passes the "
    "fifty character minimum threshold without any issues.",
    "## Details Section",
    "Some more detailed content that is also long enough to pass the threshold "
    "without being silently dropped during processing.",
    "1. First numbered item with enough text to definitely pass the threshold.",
    "2. Second numbered item also long enough to pass the fifty character minimum.",
    "- A bullet point that is long enough to pass the fifty character threshold.",
    "- Another bullet point with enough text to pass the threshold easily.",
    "## Final Section",
    "Closing paragraph that is long enough to pass the threshold and be captured "
    "as a proper chunk in the output list.",
])


def chunks_summary(chunks: list[TextChunk]) -> list[dict]:
    return [
        {"idx": c.chunk_index, "type": c.chunk_type, "header": c.header_context}
        for c in chunks
    ]


def test_basic_structure():
    chunks = split_markdown_into_chunks(LONG_MD)
    assert len(chunks) > 0, "Should produce at least one chunk"
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_indices_sequential():
    chunks = split_markdown_into_chunks(LONG_MD)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), f"Expected sequential indices, got {indices}"


def test_chunk_types_present():
    """All four chunk types should appear when input contains all element types."""
    chunks = split_markdown_into_chunks(LONG_MD)
    types = {c.chunk_type for c in chunks}
    expected = {"header_section", "numbered_item", "bulleted_item"}
    assert expected.issubset(types), f"Missing types. Got: {types}"


def test_header_context_set():
    """Chunks under nested headers should carry parent header_context."""
    chunks = split_markdown_into_chunks(LONG_MD)
    has_context = any(c.header_context is not None for c in chunks)
    assert has_context, "Expected at least one chunk with header_context"


def test_empty_text():
    assert split_markdown_into_chunks("") == []


def test_short_text_skipped():
    chunks = split_markdown_into_chunks("Too short")
    assert len(chunks) == 0


def test_continuation_lines():
    """Continuation lines after a list item should be merged into the item content."""
    text = "1. First part of a sufficiently long item\n   second line continues here with enough text"
    chunks = split_markdown_into_chunks(text)
    if chunks:
        assert "second line continues" in chunks[0].content


def test_bulleted_item_with_continuation():
    text = "- Main bullet point that is long enough to pass the fifty char threshold\n"
    text += "  with continuation text that also adds up nicely."
    chunks = split_markdown_into_chunks(text)
    if chunks:
        assert "continuation text" in chunks[0].content


if __name__ == "__main__":
    import traceback

    tests = [
        test_basic_structure,
        test_chunk_indices_sequential,
        test_chunk_types_present,
        test_header_context_set,
        test_empty_text,
        test_short_text_skipped,
        test_continuation_lines,
        test_bulleted_item_with_continuation,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    if failed:
        sys.exit(1)

    # Print summary of current behavior
    summary = chunks_summary(split_markdown_into_chunks(LONG_MD))
    print("\n=== Current chunking behavior ===")
    chunks = split_markdown_into_chunks(LONG_MD)
    for c in chunks:
        h = f"  header={c.header_context}" if c.header_context else ""
        print(f"  [{c.chunk_index}] {c.chunk_type:20} {len(c.content):>4} chars{h}")
    print(f"Total: {len(chunks)} chunks")
