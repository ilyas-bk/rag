from typing import Callable
import os

from src import MAX_CHARS


def split_by_lines(
    text: str,
    max_chars: int,
    is_boundary: Callable[[str], bool],
    min_fill: float = 0.5,
) -> list[dict]:
    """Group lines into chunks of at most max_chars.

    A new chunk starts when adding the line would overflow, or when the line
    is a "natural boundary" (def/class, heading, blank line...) and the
    current chunk is already at least min_fill full. A single line longer
    than max_chars is cut into max_chars-sized pieces.
    """
    chunks = []
    chunk_start = 0
    pos = 0

    for line in text.splitlines(keepends=True):
        size = pos - chunk_start
        if size > 0 and (
            size + len(line) > max_chars
            or (is_boundary(line) and size >= max_chars * min_fill)
        ):
            chunks.append(
                {"start": chunk_start, "end": pos, "text": text[chunk_start:pos]}
            )
            chunk_start = pos
        while pos + len(line) - chunk_start > max_chars:
            cut = chunk_start + max_chars
            chunks.append(
                {"start": chunk_start, "end": cut, "text": text[chunk_start:cut]}
            )
            chunk_start = cut
        pos += len(line)

    if chunk_start < len(text):
        chunks.append(
            {"start": chunk_start, "end": len(text), "text": text[chunk_start:]}
        )
    return chunks


def split_python_code(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split python files primarily by top-level blocks (def/class)."""
    return split_by_lines(
        text,
        max_chars,
        lambda line: line.startswith(("def ", "class ", "async def ", "@")),
    )


def split_markdown(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split markdown files by headings to keep document sections together."""
    return split_by_lines(
        text,
        max_chars,
        lambda line: line.startswith("#"),
        min_fill=0.25,
    )


def split_plain_text(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split regular text files by paragraphs (blank lines)."""
    return split_by_lines(text, max_chars, lambda line: line.strip() == "")


def chunk_file(args: tuple[str, int]) -> list[dict]:
    """Router function that sends files to the correct splitter based on extension."""
    file_path, max_chars = args
    _, ext = os.path.splitext(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            content = f.read()
    except OSError as e:
        print(f"Skipping {file_path}: {e}")
        return []

    if not content.strip():
        return []

    if ext == ".py":
        raw_chunks = split_python_code(content, max_chars)
    elif ext == ".md":
        raw_chunks = split_markdown(content, max_chars)
    else:
        raw_chunks = split_plain_text(content, max_chars)

    return [
        {"file": file_path, **chunk} for chunk in raw_chunks if chunk["text"].strip()
    ]
