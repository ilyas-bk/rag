"""Indexing: build and persist the BM25 index of the corpus."""

import json
import os
from collections import Counter, defaultdict
from typing import TypedDict

from tqdm import tqdm

from src import MAX_CHARS, INDEX_DIR, REPO_PATH
from .chunk import chunk_file
from src.utils import tokenize


class BM25Index(TypedDict):
    """Serialisable BM25 inverted index.

    Attributes
    ----------
    N : int
        Number of indexed chunks.
    avgdl : float
        Average chunk length, in terms.
    lengths : list of int
        Length of every chunk, in terms, indexed by chunk id.
    postings : dict
        Maps a term to ``{chunk_id: term frequency}``.
    """

    N: int
    avgdl: float
    lengths: list[int]
    postings: dict[str, dict[int, int]]


def build_bm25(chunks: list[dict[str, str]]) -> BM25Index:
    """Build an inverted index: term -> {chunk_id: term frequency}.

    The file path of each chunk is indexed along with its text.

    Parameters
    ----------
    chunks : list of dict
        Chunks with at least the keys ``text`` and ``file``.

    Returns
    -------
    BM25Index
        The index, ready to be dumped to JSON.
    """
    postings: dict[str, dict[int, int]] = defaultdict(dict)
    lengths: list[int] = []

    for i, chunk in enumerate(tqdm(chunks, desc="Tokenizing", unit=" chunks")):
        terms = tokenize(chunk["text"] + " " + chunk["file"])
        lengths.append(len(terms))
        for term, tf in Counter(terms).items():
            postings[term][i] = tf

    avgdl = sum(lengths) / len(lengths) if lengths else 0.0

    return {
        "N": len(chunks),
        "avgdl": avgdl,
        "lengths": lengths,
        "postings": dict(postings),
    }


def build_repository_index(
    max_chars: int = MAX_CHARS,
    repo_path: str = REPO_PATH,
    index_dir: str = INDEX_DIR,
) -> None:
    """Chunk the repository and persist the BM25 index.

    Every ``.py``, ``.md`` and ``.txt`` file under ``repo_path`` is
    chunked, then ``chunks.json`` and ``bm25.json`` are written to
    ``index_dir``.

    Parameters
    ----------
    max_chars : int, default MAX_CHARS
        Maximum number of characters of a chunk.
    repo_path : str, default REPO_PATH
        Root of the corpus to ingest.
    index_dir : str, default INDEX_DIR
        Directory where the index files are written.

    Raises
    ------
    ValueError
        If ``repo_path`` is not a directory or yields no chunk.
    """
    # Get valid files.
    file_paths: list[str] = []
    if not os.path.isdir(repo_path):
        raise ValueError(
            f"Couldn't find repository folder in `{repo_path}`!"
            " make sure it's created"
        )

    for root, _, files in os.walk(repo_path):
        for filename in files:
            path = os.path.join(root, filename)
            if path.endswith((".py", ".md", ".txt")):
                file_paths.append(path)

    chunks: list[dict[str, str]] = []
    for file_path in tqdm(file_paths, desc="Chunking", unit=" files"):
        chunks.extend(chunk_file((file_path, max_chars)))

    if not chunks:
        raise ValueError("No Chunks found! check repository files")

    bm25 = build_bm25(chunks)

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "chunks.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    with open(os.path.join(index_dir, "bm25.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(bm25, f, indent=2)

    print("Ingestion complete! Indexed "
          f"{len(chunks)} chunks under {index_dir}/")


if __name__ == "__main__":
    build_repository_index()
