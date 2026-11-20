import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm

from src import MAX_CHARS, INDEX_DIR, REPO_PATH
from .chunk import chunk_file

from src.utils import tokenize


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


# Build + save (action)


def build_repository_index(
    max_chars: int = MAX_CHARS, repo_path: str = REPO_PATH, index_dir: str = INDEX_DIR
) -> None:
    # get valid files
    file_paths = []
    if not os.path.isdir(repo_path):
        raise ValueError(
            f"Couldn't find repository folder in `{repo_path}`! make sure it's created"
        )

    for root, _, files in os.walk(repo_path):
        for f in files:
            path = os.path.join(root, f)
            if path.endswith((".py", ".md", ".txt")):
                file_paths.append(path)

    results = []
    for file_path in tqdm(file_paths, desc="Chunking", unit=" files"):
        results.append(chunk_file((file_path, max_chars)))
    chunks = [chunk for file_chunks in results for chunk in file_chunks]

    if not chunks:
        raise ValueError("No Chunks found! check repository files")

    bm25 = build_bm25(chunks)

    os.makedirs(index_dir, exist_ok=True)
    # save raw chunks
    with open(os.path.join(index_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)
    # save bm25 data
    with open(os.path.join(index_dir, "bm25.json"), "w") as f:
        json.dump(bm25, f, indent=2)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under {index_dir}/")


if __name__ == "__main__":
    build_repository_index()
