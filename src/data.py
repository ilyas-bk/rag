"""Pydantic models exchanged between the stages of the RAG pipeline."""

import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """A source location: a character range inside a corpus file.

    Attributes
    ----------
    file_path : str
        Path of the file, written exactly as in the ingested corpus.
    first_character_index : int
        Index of the first character of the range.
    last_character_index : int
        Index of the end of the range.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question that has not been answered yet.

    Attributes
    ----------
    question_id : str
        Unique identifier, a random UUID4 when not provided.
    question : str
        The text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer.

    Attributes
    ----------
    sources : list of MinimalSource
        Locations that contain the answer.
    answer : str
        The reference answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of answered and/or unanswered questions.

    Attributes
    ----------
    rag_questions : list of AnsweredQuestion or UnansweredQuestion
        The questions of the dataset.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question.

    Attributes
    ----------
    question_id : str
        Identifier of the question.
    question : str
        The text of the question.
    retrieved_sources : list of MinimalSource
        Retrieved sources, best first.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources together with the generated answer.

    Attributes
    ----------
    answer : str
        The answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset.

    Attributes
    ----------
    search_results : list of MinimalSearchResults
        One entry per question.
    k : int
        Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with generated answers for a whole dataset.

    Attributes
    ----------
    search_results : list of MinimalAnswer
        One entry per question.
    k : int
        Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
