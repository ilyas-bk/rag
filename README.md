*This project has been created as part of the 42 curriculum by ibouelba.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about a codebase. It indexes the [vLLM 0.10.1](https://github.com/vllm-project/vllm) repository, retrieves the most relevant snippets for a question with BM25, and generates a grounded answer with a small local model (`Qwen/Qwen3-0.6B`).

## Table of contents

- [Description](#description)
- [Instructions](#instructions)
- [Example usage](#example-usage)
- [System architecture](#system-architecture)
- [Chunking strategy](#chunking-strategy)
- [Retrieval method](#retrieval-method)
- [Answer generation](#answer-generation)
- [Performance analysis](#performance-analysis)
- [Design decisions](#design-decisions)
- [Challenges faced](#challenges-faced)
- [Project structure](#project-structure)
- [Resources](#resources)

## Description

Language models only know what they saw during training. Asking one about a specific codebase leads to guesses or hallucinations. RAG fixes this by looking up the relevant evidence at question time and handing it to the model.

This project implements the full pipeline end to end:

1. **Index** the vLLM source tree (`.py`, `.md`, `.txt` files) into character-ranged chunks and build a BM25 inverted index.
2. **Retrieve** the top-k chunks for a question, returned as `file_path [first_char:last_char]` source locations.
3. **Generate** a short answer from the retrieved chunks with `Qwen/Qwen3-0.6B`.
4. **Evaluate** retrieval quality with recall@k (a source counts as found when a retrieved chunk is in the same file and overlaps it with IoU > 0.05).

Everything is driven by a [Python Fire](https://github.com/google/python-fire) CLI, and all data exchanged between stages is validated with `pydantic` models.

## Instructions

### Requirements

- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The vLLM 0.10.1 sources in `data/raw/vllm-0.10.1/` (already part of the repo layout)
- Enough memory to load `Qwen/Qwen3-0.6B` (only needed for the `answer*` commands; a GPU is used automatically if available)

### Installation

```bash
git clone https://github.com/ilyas-bk/rag.git
cd rag
make install        # runs `uv sync`
```

### Commands

Every command is invoked as `uv run python -m src <command> [options]`. The Makefile wraps them with sensible defaults.

| Command | Description | Makefile target |
|---|---|---|
| `index --max_chunk_size <int>` | Chunk `data/raw/` and build the index under `data/processed/` | `make index` |
| `search <query> --k <int>` | Print the top-k sources for one query | `make search query="..." k=5` |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Search a whole dataset and write a `StudentSearchResults` JSON | `make search_dataset` |
| `answer <query> --k <int>` | Answer a single query from the retrieved context | `make answer query="..."` |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Generate answers for a saved search-results file | `make answer_dataset` |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Compute recall@1/3/5/10 against a ground-truth dataset | `make evaluate` |

Makefile variables can be overridden on the command line, for example `make search_dataset file_type=code k=10`.

### Linting

```bash
make lint           # flake8 + mypy
```

## Example usage

**1. Build the index** (`--max_chunk_size` must be between 1 and 2000, default 2000):

```console
$ uv run python -m src index --max_chunk_size 2000
Chunking:   100%|##########| 1965/1965 [00:00<00:00, 13205 files/s]
Tokenizing: 100%|##########| 12572/12572 [00:02<00:00, 5796 chunks/s]
Ingestion complete! Indexed 12572 chunks under data/processed/
```

**2. Search a single query:**

```console
$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/docs/... [start:end]
...
```

**3. Search a whole dataset and evaluate it:**

```console
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

**4. Generate answers from the saved search results:**

```console
$ uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The official recall@k is computed by the provided `moulinette` executable (in `moulinette/`), which the code never imports or calls:

```bash
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```

Invalid input (empty query, `k=0`, missing files, malformed JSON, missing index) prints an error message instead of crashing.

## System architecture

```
 data/raw/vllm-0.10.1            INDEXING (src/index)
 (.py / .md / .txt)  ───────►  chunk_file ──► chunks.json ─┐
                                 │                         ├─► data/processed/
                                 └─► tokenize ─► bm25.json ┘
                                                              │
 question ──► tokenize ──► BM25Retriever (src/retrieve) ◄─────┘
                                 │
                                 ▼
                      top-k MinimalSource (file, start, end)
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
     StudentSearchResults JSON         Generate (src/generate)
     ──► evaluate / moulinette         read chunk text ──► prompt ──► Qwen3-0.6B
                                                  │
                                                  ▼
                                   StudentSearchResultsAndAnswer JSON
```

| Component | Module | Role |
|---|---|---|
| CLI | `src/cli.py`, `src/__main__.py` | Fire entry point, input validation, dataset loops, evaluation |
| Data models | `src/data.py` | `pydantic` models for sources, questions, search results and answers |
| Chunking | `src/index/chunk.py` | File-type-aware splitting into character ranges |
| Indexing | `src/index/index.py` | Walks the corpus, builds and saves the BM25 index |
| Tokenizer | `src/utils.py` | Code-aware tokenizer shared by indexing and querying |
| Retrieval | `src/retrieve/retrieve.py` | `BM25Retriever`: loads the index and ranks chunks |
| Generation | `src/generate/generate.py` | Builds the prompt from retrieved text and runs Qwen3-0.6B |

## Chunking strategy

Chunks are stored as `{file, start, end, text}`, where `start`/`end` are character offsets in the original file. This is exactly what the grader compares, so the index and the output share the same representation. Files are read with `newline=""` so offsets are not shifted by newline translation.

Chunking is line-based (`split_by_lines`): lines are accumulated until adding the next one would exceed `max_chunk_size` (never more than 2000 characters). A chunk is also closed early at a "natural boundary" once it is at least `min_fill` full. A single line longer than the limit is hard-cut into pieces. The boundary depends on the file type:

| File type | Strategy | Boundary | `min_fill` |
|---|---|---|---|
| `.py` | Python code | Lines starting with `def `, `class `, `async def ` or `@` (top-level definitions and decorators) | 0.5 |
| `.md` | Markdown | Lines starting with `#` (headings keep a section together) | 0.25 |
| `.txt` | Plain text | Blank lines (paragraphs) | 0.5 |

Empty or whitespace-only chunks are dropped.

## Retrieval method

Retrieval uses **BM25** implemented from scratch (no external search library), with `k1 = 1.5` and `b = 0.75`:

```
score(q, d) = Σ_t  idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))
idf(t)      = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))
```

- The index is an **inverted index** (`term → {chunk_id: tf}`) plus per-chunk lengths, so a query only touches the chunks that contain its terms.
- A **code-aware tokenizer** (`src/utils.py`) keeps the full identifier *and* its sub-words, so both a verbatim identifier and a natural-language paraphrase can match:
  - `load_lora_adapter` → `load_lora_adapter`, `load`, `lora`, `adapter`
  - `FusedMoELayer` → `fusedmoelayer`, `fused`, `mo`, `e`, `layer`
- The **file path is indexed together with the chunk text**, so a question mentioning a module or feature name can match on the path.
- Query terms are de-duplicated, scored per chunk, and the top-k are returned best first.

## Answer generation

`Generate` loads `Qwen/Qwen3-0.6B` with `transformers` (bfloat16, `device_map="auto"`). For each question it:

1. reads the text of each retrieved source range from disk,
2. concatenates the snippets as numbered `Context Block i` sections inside a system prompt,
3. instructs the model to answer **only** from the context, without outside knowledge, and to reply `couldn't find the answer in the context` when the evidence is insufficient,
4. runs the chat template with thinking disabled and `max_new_tokens=256`, then decodes the answer.

The strict prompt trades completeness for faithfulness, which suits a 0.6B model that is prone to hallucination.

## Performance analysis

Measured on the public datasets (`data/datasets/AnsweredQuestions/`), with the default chunk size of 2000, BM25, and k = 10:

| Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|---|
| docs | 100 | 0.630 | 0.780 | **0.820** | 0.890 | 80 % |
| code | 99 | 0.566 | 0.768 | **0.838** | 0.879 | 50 % |

| Requirement | Limit | Measured |
|---|---|---|
| Indexing the whole corpus | ≤ 5 min | ~3 s (1965 files, 12 572 chunks) |
| Retrieval throughput | ≤ 90 s for 200 questions | ~1.2 s for ~100 questions |

Both recall@5 thresholds are met. Recall keeps improving up to k = 10, which shows the right chunk is usually retrieved but not always ranked first. That is the typical limitation of a purely lexical method. Figures can vary slightly with the chunk size and the exact corpus snapshot.

## Design decisions

- **BM25 over TF-IDF**: length normalisation matters because chunks vary in size, and BM25 saturates term frequency, so a chunk that repeats a word many times does not dominate.
- **From-scratch index**: a plain JSON inverted index is simple, fast enough for this corpus, and easy to inspect (`chunks.json` and `bm25.json` under `data/processed/`).
- **Code-aware tokenization**: source code mixes `snake_case`, `CamelCase` and prose. Indexing both the whole identifier and its parts is what lets exact identifiers and paraphrased questions both match.
- **Structure-aware chunk boundaries**: splitting at definitions and headings keeps related code or documentation together, while the size cap guarantees no source exceeds the grader's 2000-character limit.
- **Character offsets as the single source of truth**: the same `(file, start, end)` triple is used for chunks, search results and evaluation.
- **Strict generation prompt**: the model must answer from the context or say it cannot, to limit hallucination.
- **Defensive CLI**: every entry point validates its input and reports errors instead of raising.

## Challenges faced

- **Matching code and natural language**: questions paraphrase ideas, while the code contains identifiers. The sub-word tokenizer and path indexing bridge that gap without embeddings.
- **Respecting the 2000-character cap**: a single oversized source invalidates the whole output, so over-long lines are hard-cut and the chunk size is validated in the CLI.
- **Exact offsets**: recall is computed on character ranges, so files are read without newline translation to keep offsets consistent with the grader.
- **Small-model limits**: Qwen3-0.6B reasons poorly over long contexts, so the prompt is restrictive and thinking mode is disabled.

## Project structure

```
.
├── Makefile
├── pyproject.toml / uv.lock
├── moulinette/                  # official grader executable
├── rag_against_the_machine.md   # project subject
├── src/
│   ├── __main__.py              # entry point (python -m src)
│   ├── cli.py                   # Fire CLI
│   ├── data.py                  # pydantic models
│   ├── utils.py                 # code-aware tokenizer
│   ├── index/                   # chunking + BM25 index building
│   ├── retrieve/                # BM25Retriever
│   └── generate/                # Qwen3-0.6B answer generation
└── data/
    ├── raw/vllm-0.10.1/         # indexed corpus
    ├── processed/               # chunks.json, bm25.json
    ├── datasets/                # AnsweredQuestions / UnansweredQuestions
    └── output/                  # search results and answers
```

## Resources

**References**

- Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401)
- Robertson & Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.nowpublishers.com/article/Details/INR-019)
- Manning, Raghavan & Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) (inverted indexes, TF-IDF, BM25, evaluation)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [vLLM](https://github.com/vllm-project/vllm), the indexed codebase
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [uv](https://docs.astral.sh/uv/)

**Use of AI**

AI tools were used for two parts of this project:

Answer generation phase (src/generate/): AI helped me work out how to load and run Qwen/Qwen3-0.6B with transformers (chat template, disabling thinking mode, decoding only the newly generated tokens) and how to write the system prompt. The prompt needed to be strict, so the small model answers only from the retrieved context and says so when the context is insufficient. I then iterated on the prompt and generation settings myself.
Documentation: AI produced a first draft of this README.md from the project's code and subject, which I then reviewed and corrected.

The indexing, chunking, BM25 retrieval, tokenizer and CLI were written by me without AI generation.

All AI-assisted output was checked before being kept. I read and tested the generation code on real questions, and the figures in the performance section come from running the pipeline on the public datasets, not from the AI's own claims. I can explain every part of the code and documentation in this repository.