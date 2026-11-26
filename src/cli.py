"""Command-line interface of the RAG system, built with Fire."""

import json
import os

from pydantic import ValidationError
from tqdm import tqdm

from src import MAX_CHARS
from src.data import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    MinimalAnswer,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.index import build_repository_index
from src.retrieve import BM25Retriever
from src.generate import Generate  # this takes time to import


def to_source(chunk: dict) -> MinimalSource:
    """Convert an indexed chunk into a :class:`MinimalSource`.

    Parameters
    ----------
    chunk : dict
        Chunk with the keys ``file``, ``start`` and ``end``.

    Returns
    -------
    MinimalSource
        The source location covered by the chunk.
    """
    return MinimalSource(
        file_path=chunk["file"],
        first_character_index=chunk["start"],
        last_character_index=chunk["end"],
    )


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute the intersection over union of two character ranges.

    Parameters
    ----------
    a : MinimalSource
        First source.
    b : MinimalSource
        Second source.

    Returns
    -------
    float
        The IoU in ``[0, 1]``; 0.0 if the files differ or the ranges do
        not overlap.
    """
    if a.file_path != b.file_path:
        return 0.0
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    union = max(a.last_character_index, b.last_character_index) - min(
        a.first_character_index, b.first_character_index
    )
    return max(inter, 0) / union if union > 0 else 0.0


class CLI:
    """Commands exposed by Python Fire: ``python -m src <command>``."""

    def index(self, max_chunk_size: int = MAX_CHARS) -> None:
        """Chunk ``data/raw/`` and build the BM25 index.

        The index is written under ``data/processed/``.

        Parameters
        ----------
        max_chunk_size : int, default MAX_CHARS
            Maximum chunk size in characters, between 1 and MAX_CHARS.
        """
        if not isinstance(max_chunk_size, int) or \
                not 0 < max_chunk_size <= MAX_CHARS:
            print("Error: --max_chunk_size must be an "
                  f"integer in 1..{MAX_CHARS}")
            return
        build_repository_index(max_chunk_size)

    def search(self, query: str = "", k: int = 5) -> None:
        """Print the top-k sources for a single query.

        Parameters
        ----------
        query : str
            The question to search for.
        k : int, default 5
            Number of sources to print.
        """
        if not str(query).strip():
            print("Error: the query is empty")
            return
        try:
            retriever = BM25Retriever()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        results = retriever.search(str(query), k)
        if not results:
            print("No results")
        for chunk in results:
            print(f"{chunk['file']} [{chunk['start']}:{chunk['end']}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search every question of a dataset and save the results.

        The output is a ``StudentSearchResults`` JSON file named like the
        dataset file.

        Parameters
        ----------
        dataset_path : str
            Path of the JSON dataset of questions.
        k : int, default 10
            Number of sources retrieved per question.
        save_directory : str, default "data/output/search_results"
            Directory where the output file is written.
        """
        try:
            with open(dataset_path) as f:
                dataset = RagDataset.model_validate(json.load(f))
            retriever = BM25Retriever()
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: could not load {dataset_path}: {e}")
            return

        search_results = []
        for q in tqdm(dataset.rag_questions,
                      desc="Searching",
                      unit=" questions"):
            chunks = retriever.search(q.question, k)
            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[to_source(c) for c in chunks],
                )
            )

        output = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory,
                                 os.path.basename(dataset_path))
        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str = "", k: int = 5) -> None:
        """Answer a single query using the retrieved context.

        Parameters
        ----------
        query : str
            The question to answer.
        k : int, default 5
            Number of sources retrieved and given to the model.
        """
        if not str(query).strip():
            print("Error: the query is empty")
            return
        try:
            retriever = BM25Retriever()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        sources = [to_source(c) for c in retriever.search(str(query), k)]
        if not sources:
            print("No results")
            return
        generator = Generate()
        print(generator.generate_response(str(query), k, sources))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset of search results.

        The output is a ``StudentSearchResultsAndAnswer`` JSON file named
        like the input file.

        Parameters
        ----------
        student_search_results_path : str
            Path of a ``StudentSearchResults`` JSON file.
        save_directory : str
            Directory where the output file is written.
        """
        try:
            with open(student_search_results_path) as f:
                results = StudentSearchResults.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: could not load {student_search_results_path}: {e}")
            return

        generator = Generate()
        search_results = []
        for r in tqdm(results.search_results,
                      desc="Generating",
                      unit=" questions"):
            answer = generator.generate_response(
                r.question,
                k=len(r.retrieved_sources),
                sources=r.retrieved_sources
            )
            search_results.append(
                MinimalAnswer(
                    question_id=r.question_id,
                    question=r.question,
                    answer=answer or "",
                    retrieved_sources=r.retrieved_sources,
                )
            )
        output = StudentSearchResultsAndAnswer(
            search_results=search_results,
            k=results.k
        )
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(
            save_directory, os.path.basename(student_search_results_path)
        )
        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        iou_threshold: float = 0.05,
    ) -> None:
        """Report recall@1/3/5/10 of saved search results.

        Results are compared with a ground-truth dataset; a source counts
        as found when a retrieved one overlaps it with an IoU above the
        threshold.

        Parameters
        ----------
        student_search_results_path : str
            Path of a ``StudentSearchResults`` JSON file.
        dataset_path : str
            Path of the ``AnsweredQuestions`` ground-truth dataset.
        iou_threshold : float, default 0.05
            Minimum IoU for a retrieved source to match an expected one.
        """
        try:
            with open(student_search_results_path) as f:
                results = StudentSearchResults.model_validate(json.load(f))
            with open(dataset_path) as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: {e}")
            return

        truth = {
            q.question_id: q.sources
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        evaluated = [r for r in results.search_results
                     if truth.get(r.question_id)]
        if not evaluated:
            print("Error: no question in common with an answered dataset")
            return

        print(f"Questions evaluated: {len(evaluated)}")
        for k in (1, 3, 5, 10):
            total = 0.0
            for r in evaluated:
                expected = truth[r.question_id]
                top = r.retrieved_sources[:k]
                found = sum(
                    any(iou(src, got) > iou_threshold for got in top)
                    for src in expected
                )
                total += found / len(expected)
            print(
                f"Recall@{k}: {total / len(evaluated):.3f} "
                f"({100 * total / len(evaluated):.1f}%)"
            )
