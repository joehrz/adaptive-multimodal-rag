"""
Test script for Parallel Self-RAG reflection
Compares sequential vs parallel execution times
"""

import time
import sys
sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

import pytest
from langchain.schema import Document

try:
    import ollama
    ollama.list()
    OLLAMA_RUNNING = True
except Exception:
    OLLAMA_RUNNING = False


@pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
def test_parallel_self_rag():
    """Test parallel vs sequential Self-RAG reflection"""
    from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG

    self_rag = OllamaSelfRAG(verbose=True)

    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
            metadata={"source": "ml_basics"}
        ),
        Document(
            page_content="Deep learning is a type of machine learning that uses neural networks with multiple layers. It excels at tasks like image recognition and natural language processing.",
            metadata={"source": "dl_intro"}
        ),
    ]

    query = "What is machine learning?"
    answer = "Machine learning is a branch of AI that allows systems to learn from data and make predictions."

    # Test sequential reflection
    start_seq = time.time()
    result_seq = self_rag.reflect_on_answer(
        query=query,
        answer=answer,
        documents=documents,
        parallel=False
    )
    time_seq = time.time() - start_seq

    # Test parallel reflection
    start_par = time.time()
    result_par = self_rag.reflect_on_answer(
        query=query,
        answer=answer,
        documents=documents,
        parallel=True
    )
    time_par = time.time() - start_par

    # Both modes should produce valid scores
    assert 0 <= result_seq.overall_score <= 1, f"Sequential score out of range: {result_seq.overall_score}"
    assert 0 <= result_par.overall_score <= 1, f"Parallel score out of range: {result_par.overall_score}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
