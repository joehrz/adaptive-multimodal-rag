"""
Evaluation module for Adaptive Multimodal RAG

Provides LLM-as-judge evaluation metrics and benchmark runners
for measuring RAG pipeline quality.
"""

from src.evaluation.metrics import RAGEvaluator, EvaluationResult, RougeScores
from src.evaluation.benchmark import BenchmarkRunner, BenchmarkCase, BenchmarkReport

__all__ = [
    "RAGEvaluator",
    "EvaluationResult",
    "RougeScores",
    "BenchmarkRunner",
    "BenchmarkCase",
    "BenchmarkReport",
]
