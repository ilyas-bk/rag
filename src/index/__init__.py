"""Indexing stage: chunking and BM25 index construction."""

from .index import build_repository_index

__all__ = ["build_repository_index"]
