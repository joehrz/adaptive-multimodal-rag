"""
HyDE data models - dataclasses for HyDE query and retrieval results.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from langchain.schema import Document


@dataclass
class HyDEResult:
    """Result from HyDE query"""
    query: str
    hypothetical_document: str
    answer: str
    retrieved_docs: List[Document]
    hyde_retrieval_count: int
    standard_retrieval_count: int
    total_time: float
    hyde_generation_time: float
    retrieval_time: float
    answer_generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "hypothetical_document": self.hypothetical_document[:500] + "..." if len(self.hypothetical_document) > 500 else self.hypothetical_document,
            "answer": self.answer,
            "hyde_retrieval_count": self.hyde_retrieval_count,
            "standard_retrieval_count": self.standard_retrieval_count,
            "total_time": self.total_time,
            "hyde_generation_time": self.hyde_generation_time,
            "retrieval_time": self.retrieval_time,
            "answer_generation_time": self.answer_generation_time
        }


@dataclass
class HyDERetrievalResult:
    """Result from HyDE retrieval-only operation (no answer generation)"""
    query: str
    hypothetical_document: str
    documents: List[Document]
    retrieval_time: float
    hyde_generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "hypothetical_document": self.hypothetical_document[:500] + "..." if len(self.hypothetical_document) > 500 else self.hypothetical_document,
            "document_count": len(self.documents),
            "retrieval_time": self.retrieval_time,
            "hyde_generation_time": self.hyde_generation_time
        }
