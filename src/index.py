import json
import os
import re
from collections import Counter, defaultdict
from typing import Callable
from tqdm import tqdm

from text_ext import is_text_file


REPO_PATH = "data/raw/vllm-0.10.1"
INDEX_DIR = "data/processed"
MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# 1. Chunking (every chunk keeps its character offsets in the original file)
# ---------------------------------------------------------------------------

def split_by_lines(
    text: str,
    max_chars: int,
    is_boundary: Callable[[str], bool],
    min_fill: float = 0.5,
) -> list[dict]:
    """Group lines into chunks of at most ~max_chars.

    A new chunk starts when adding the line would overflow, or when the line
    is a "natural boundary" (def/class, heading, blank line...) and the
    current chunk is already at least min_fill full.
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
            chunks.append({"start": chunk_start, "end": pos,
                           "text": text[chunk_start:pos]})
            chunk_start = pos
        pos += len(line)

    if chunk_start < len(text):
        chunks.append({"start": chunk_start, "end": len(text),
                       "text": text[chunk_start:]})
    return chunks


def split_python_code(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split python files primarily by top-level blocks (def/class)."""
    return split_by_lines(
        text, max_chars,
        lambda line: line.startswith(("def ", "class ", "async def ", "@")),
    )


def split_markdown(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split markdown files by headings to keep document sections together."""
    return split_by_lines(
        text, max_chars,
        lambda line: line.startswith("#"),
        min_fill=0.25,
    )


def split_plain_text(text: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """Split regular text files by paragraphs (blank lines)."""
    return split_by_lines(text, max_chars, lambda line: line.strip() == "")


def process_file(args: tuple[str, int]) -> list[dict]:
    """Router function that sends files to the correct splitter based on extension."""
    file_path, max_chars = args
    _, ext = os.path.splitext(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore",
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
        for chunk in raw_chunks
        if chunk["text"].strip()
    ]


# ---------------------------------------------------------------------------
# 2. BM25 index
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: keeps whole identifiers and their sub-words.

    "load_lora_adapter" -> ["load_lora_adapter", "load", "lora", "adapter"]
    "FusedMoELayer"     -> ["fusedmoelayer", "fused", "mo", "e", "layer"]
    """
    terms = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        lower = word.lower()
        terms.append(lower)
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word)
        if len(parts) > 1:
            terms.extend(p.lower() for p in parts)
    return terms


def build_bm25(chunks: list[dict]) -> dict:
    """Build an inverted index: term -> {chunk_id: term frequency}."""
    postings: dict[str, dict[int, int]] = defaultdict(dict)
    lengths = []

    for i, chunk in enumerate(tqdm(chunks, desc="Tokenizing", unit=" chunks")):
        terms = tokenize(chunk["text"] + " " + chunk["file"])
        lengths.append(len(terms))
        for term, tf in Counter(terms).items():
            postings[term][i] = tf

    return {
        "N": len(chunks),
        "avgdl": sum(lengths) / len(lengths),
        "lengths": lengths,
        "postings": postings,
    }


# ---------------------------------------------------------------------------
# 3. Build + save
# ---------------------------------------------------------------------------

def build_repository_index(max_chars: int = MAX_CHARS,
                           repo_path: str = REPO_PATH,
                           index_dir: str = INDEX_DIR) -> None:
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            path = os.path.join(root, f)
            if is_text_file(path):
                file_paths.append(path)

    results = []
    for file_path in tqdm(file_paths, desc="Chunking", unit=" files"):
        results.append(process_file((file_path, max_chars)))
    chunks = [chunk for file_chunks in results for chunk in file_chunks]

    bm25 = build_bm25(chunks)

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)
    with open(os.path.join(index_dir, "bm25.json"), "w") as f:
        json.dump(bm25, f, indent=2)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {index_dir}/")


if __name__ == "__main__":
    build_repository_index()

