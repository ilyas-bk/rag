"""Chunking strategies: Python code, Markdown and plain text."""

from typing import Callable
import os

from src import MAX_CHARS


def split_by_lines(
    text: str,
    max_chars: int,
    is_boundary: Callable[[str], bool],
    min_fill: float = 0.5,
) -> list[dict]:
    """Group lines into chunks of at most ``max_chars`` characters.

    A new chunk starts when adding the line would overflow, or when the
    line is a "natural boundary" (def/class, heading, blank line...) and
    the current chunk is already at least ``min_fill`` full. A single
    line longer than ``max_chars`` is cut into ``max_chars``-sized
    pieces.

    Parameters
    ----------
    text : str
        The full text to split.
    max_chars : int
        Maximum number of characters of a chunk.
    is_boundary : callable
        Takes a line and returns True if a new chunk may start there.
    min_fill : float, default 0.5
        Fraction of ``max_chars`` a chunk must reach before a boundary
        line is allowed to start a new chunk.

    Returns
    -------
    list of dict
        Chunks with the keys ``start`` and ``end`` (character range in
        ``text``) and ``text``.
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
                {"start": chunk_start,
                 "end": pos,
                 "text": text[chunk_start:pos]}
            )
            chunk_start = pos
        while pos + len(line) - chunk_start > max_chars:
            cut = chunk_start + max_chars
            chunks.append(
                {"start": chunk_start,
                 "end": cut,
                 "text": text[chunk_start:cut]}
            )
            chunk_start = cut
        pos += len(line)

    if chunk_start < len(text):
        chunks.append(
            {"start": chunk_start,
             "end": len(text),
             "text": text[chunk_start:]}
        )
    return chunks


def split_python_code(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split Python source primarily by top-level blocks (def/class).

    Parameters
    ----------
    text : str
        The content of a Python file.
    max_chars : int, default MAX_CHARS
        Maximum number of characters of a chunk.

    Returns
    -------
    list of dict
        Chunks with the keys ``start``, ``end`` and ``text``.
    """
    return split_by_lines(
        text,
        max_chars,
        lambda line: line.startswith(("def ", "class ", "async def ", "@")),
    )


def split_markdown(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split Markdown by headings to keep document sections together.

    Parameters
    ----------
    text : str
        The content of a Markdown file.
    max_chars : int, default MAX_CHARS
        Maximum number of characters of a chunk.

    Returns
    -------
    list of dict
        Chunks with the keys ``start``, ``end`` and ``text``.
    """
    return split_by_lines(
        text,
        max_chars,
        lambda line: line.startswith("#"),
        min_fill=0.25,
    )


def split_plain_text(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split plain text by paragraphs (blank lines).

    Parameters
    ----------
    text : str
        The content of a text file.
    max_chars : int, default MAX_CHARS
        Maximum number of characters of a chunk.

    Returns
    -------
    list of dict
        Chunks with the keys ``start``, ``end`` and ``text``.
    """
    return split_by_lines(text, max_chars, lambda line: line.strip() == "")


def chunk_file(args: tuple[str, int]) -> list[dict]:
    """Read a file and split it with the splitter fitting its extension.

    ``.py`` files use :func:`split_python_code`, ``.md`` files use
    :func:`split_markdown` and any other file is split as plain text.
    Unreadable and blank files are skipped.

    Parameters
    ----------
    args : tuple of (str, int)
        The file path and the maximum chunk size in characters.

    Returns
    -------
    list of dict
        Non-blank chunks with the keys ``file``, ``start``, ``end`` and
        ``text``. Empty if the file cannot be read or is blank.
    """
    file_path, max_chars = args
    _, ext = os.path.splitext(file_path)

    try:
        with open(file_path, "r",
                  encoding="utf-8",
                  errors="ignore",
                  newline="") as f:
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
        {"file": file_path, **chunk}
        for chunk in raw_chunks if chunk["text"].strip()
    ]
