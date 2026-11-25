import json
import math
import os

from src import INDEX_DIR
from src.utils import tokenize


class BM25Retriever:
    """Loads the index built by `index` and ranks chunks for a query.

    For every query term t present in chunk d, BM25 adds:

        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))

    - idf(t): rare terms ("load_lora_adapter") weigh more than common ones
    - tf: how often t appears in d, with diminishing returns (k1)
    - len(d) / avgdl: long chunks are penalised so they don't win by size (b)
    """

    def __init__(
        self, index_dir: str = INDEX_DIR, k1: float = 1.5, b: float = 0.75
    ) -> None:
        chunks_path = os.path.join(index_dir, "chunks.json")
        bm25_path = os.path.join(index_dir, "bm25.json")
        if not (os.path.isfile(chunks_path) and os.path.isfile(bm25_path)):
            raise FileNotFoundError(
                f"No index found in {index_dir}/, "
                "run the `index` command first"
            )

        with open(chunks_path) as f:
            self.chunks: list[dict] = json.load(f)
        with open(bm25_path) as f:
            bm25 = json.load(f)

        self.k1 = k1
        self.b = b
        self.avgdl: float = bm25["avgdl"]
        self.lengths: list[int] = bm25["lengths"]
        self.postings: dict[str, dict[int, int]] = {
            term: {int(cid): tf for cid, tf in docs.items()}
            for term, docs in bm25["postings"].items()
        }
        n = bm25["N"]
        self.idf: dict[str, float] = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the k best chunks for the query, best first."""
        if k <= 0:
            return []

        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            docs = self.postings.get(term)
            if docs is None:
                continue
            idf = self.idf[term]
            for cid, tf in docs.items():
                norm = 1 - self.b + self.b * self.lengths[cid] / self.avgdl
                score = idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
                scores[cid] = scores.get(cid, 0.0) + score

        best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [self.chunks[cid] for cid in best]
