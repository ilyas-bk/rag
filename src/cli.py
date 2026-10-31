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
from src.generate import generate_response


def to_source(chunk: dict) -> MinimalSource:
    return MinimalSource(
        file_path=chunk["file"],
        first_character_index=chunk["start"],
        last_character_index=chunk["end"],
    )


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character ranges in the same file."""
    if a.file_path != b.file_path:
        return 0.0
    inter = (min(a.last_character_index, b.last_character_index)
             - max(a.first_character_index, b.first_character_index))
    union = (max(a.last_character_index, b.last_character_index)
             - min(a.first_character_index, b.first_character_index))
    return max(inter, 0) / union if union > 0 else 0.0


class CLI:
    def index(self, max_chunk_size: int = MAX_CHARS) -> None:
        """Chunk data/raw/ and build the BM25 index under data/processed/."""
        if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= MAX_CHARS:
            print(f"Error: --max_chunk_size must be an integer in 1..{MAX_CHARS}")
            return
        build_repository_index(max_chunk_size)

    def search(self, query: str = "", k: int = 5) -> None:
        """Print the top-k sources for a single query."""
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
        print(results)
        for chunk in results:
            print(f"{chunk['file']} [{chunk['start']}:{chunk['end']}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        """Search every question of a dataset and save StudentSearchResults."""
        try:
            with open(dataset_path) as f:
                dataset = RagDataset.model_validate(json.load(f))
            retriever = BM25Retriever()
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: could not load {dataset_path}: {e}")
            return

        search_results = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit=" questions"):
            chunks = retriever.search(q.question, k)
            search_results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=[to_source(c) for c in chunks],
            ))

        output = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, os.path.basename(dataset_path))
        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results to {save_path}")


    def answer(self, query: str = "", k: int = 5):
        result = generate_response(query, k)
        print(result)


    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file."""
        try:
            with open(student_search_results_path) as f:
                results = StudentSearchResults.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: could not load {student_search_results_path}: {e}")
            return

        search_results = []
        for r in results.search_results:
            answer = generate_response(r.question, k=len(r.retrieved_sources))
            search_results.append(MinimalAnswer(
                question_id=r.question_id,
                question=r.question,
                retrieved_sources=r.retrieved_sources,
                answer=answer
            ))
        k = len(search_results)
        output = StudentSearchResultsAndAnswer(search_results=search_results, k=k)
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, os.path.basename(student_search_results_path))
        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {save_path}")


    def evaluate(self, student_search_results_path: str, dataset_path: str,
                 iou_threshold: float = 0.05) -> None:
        """Report recall@1/3/5/10 of saved search results vs a ground truth."""
        try:
            with open(student_search_results_path) as f:
                results = StudentSearchResults.model_validate(json.load(f))
            with open(dataset_path) as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error: {e}")
            return

        truth = {q.question_id: q.sources for q in dataset.rag_questions
                 if isinstance(q, AnsweredQuestion)}
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
            print(f"Recall@{k}: {total / len(evaluated):.3f} "
                  f"({100 * total / len(evaluated):.1f}%)")
