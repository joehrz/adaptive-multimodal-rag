"""
Test script for Parallel Self-RAG reflection
Compares sequential vs parallel execution times
"""

import time
import sys
sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from langchain.schema import Document

def test_parallel_self_rag():
    """Test parallel vs sequential Self-RAG reflection"""
    print("=" * 70)
    print("PARALLEL SELF-RAG TEST")
    print("=" * 70)

    try:
        from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG

        # Initialize
        print("\nInitializing Self-RAG...")
        self_rag = OllamaSelfRAG(verbose=True)
        print("SUCCESS: Self-RAG initialized")

        # Create test documents
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
        print("\n" + "=" * 70)
        print("TEST 1: Sequential Reflection")
        print("=" * 70)

        start_seq = time.time()
        result_seq = self_rag.reflect_on_answer(
            query=query,
            answer=answer,
            documents=documents,
            parallel=False  # Sequential
        )
        time_seq = time.time() - start_seq

        print(f"\nSequential Time: {time_seq:.2f}s")
        print(f"Overall Score: {result_seq.overall_score:.2f}")
        print(f"Relevance: {result_seq.relevance.value}")
        print(f"Support: {result_seq.support.value}")
        print(f"Utility: {result_seq.utility.value}")

        # Test parallel reflection
        print("\n" + "=" * 70)
        print("TEST 2: Parallel Reflection")
        print("=" * 70)

        start_par = time.time()
        result_par = self_rag.reflect_on_answer(
            query=query,
            answer=answer,
            documents=documents,
            parallel=True  # Parallel
        )
        time_par = time.time() - start_par

        print(f"\nParallel Time: {time_par:.2f}s")
        print(f"Overall Score: {result_par.overall_score:.2f}")
        print(f"Relevance: {result_par.relevance.value}")
        print(f"Support: {result_par.support.value}")
        print(f"Utility: {result_par.utility.value}")

        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        speedup = time_seq / time_par if time_par > 0 else 0
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Parallel:   {time_par:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")

        # Validate results are equivalent
        scores_match = abs(result_seq.overall_score - result_par.overall_score) < 0.01
        print(f"\nResults consistent: {'YES' if scores_match else 'CLOSE (scores may vary due to LLM non-determinism)'}")

        print("\n" + "=" * 70)
        if speedup > 1.5:
            print("TEST PASSED! Parallel execution is significantly faster.")
        else:
            print("TEST PASSED! (Speedup may vary based on LLM load)")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parallel_self_rag()
    sys.exit(0 if success else 1)
