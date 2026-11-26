"""RAG against the machine: question answering over the vLLM codebase.

The package is split in four stages that feed each other:

- ``index``: chunk the corpus and build a BM25 inverted index,
- ``retrieve``: rank the indexed chunks for a query,
- ``generate``: answer a question from the retrieved chunks,
- ``cli``: the Python Fire command-line interface.
"""

REPO_PATH = "data/raw/vllm-0.10.1"
INDEX_DIR = "data/processed"
MAX_CHARS = 2000
