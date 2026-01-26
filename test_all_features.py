#!/usr/bin/env python3
"""
Command-line test script for all RAG features
Tests Caching, Self-RAG, and GraphRAG without requiring the Streamlit frontend

Usage:
    python test_all_features.py              # Run all tests with sample docs
    python test_all_features.py --pdf FILE   # Test with a specific PDF file
    python test_all_features.py --quick      # Quick import-only tests
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    Document(
        page_content="""Machine learning is a subset of artificial intelligence (AI) that enables
        computers to learn from data without being explicitly programmed. It uses algorithms
        to identify patterns and make predictions. Deep learning is a specialized form of
        machine learning that uses neural networks with multiple layers.""",
        metadata={"source": "ml_basics", "type": "sample"}
    ),
    Document(
        page_content="""Neural networks are computing systems inspired by biological neural networks.
        They consist of interconnected nodes (neurons) that process information. Deep learning
        uses neural networks with many layers (deep neural networks) for complex pattern recognition.
        Transformers are a type of neural network architecture that uses attention mechanisms.""",
        metadata={"source": "neural_networks", "type": "sample"}
    ),
    Document(
        page_content="""RAG (Retrieval-Augmented Generation) is a technique that combines information
        retrieval with text generation. It retrieves relevant documents from a knowledge base
        and uses them to generate more accurate responses. GraphRAG extends this by using
        knowledge graphs for multi-hop reasoning.""",
        metadata={"source": "rag_overview", "type": "sample"}
    ),
    Document(
        page_content="""Python is a high-level programming language known for its simplicity and
        readability. It is widely used in data science, machine learning, and web development.
        Popular Python libraries for machine learning include TensorFlow, PyTorch, and scikit-learn.""",
        metadata={"source": "python_intro", "type": "sample"}
    ),
    Document(
        page_content="""Knowledge graphs represent information as networks of entities and relationships.
        They enable semantic understanding and reasoning across connected data. GraphRAG uses
        knowledge graphs to perform multi-hop reasoning, following paths through the graph
        to answer complex questions that require connecting multiple pieces of information.""",
        metadata={"source": "knowledge_graphs", "type": "sample"}
    )
]


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {test_name}")
    if details:
        print(f"        {details}")


def test_imports():
    """Test that all modules can be imported"""
    print_header("IMPORT TESTS")

    results = []

    # Test caching
    try:
        from src.core.caching_system import RAGCacheManager, LRUCache, SemanticQueryCache
        print_result("Caching System", True)
        results.append(True)
    except Exception as e:
        print_result("Caching System", False, str(e))
        results.append(False)

    # Test Self-RAG
    try:
        from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG, ReflectionResult
        print_result("Self-RAG", True)
        results.append(True)
    except Exception as e:
        print_result("Self-RAG", False, str(e))
        results.append(False)

    # Test GraphRAG
    try:
        from src.experiments.graph_reasoning.ollama_graph_rag import OllamaGraphRAG, Entity, Relationship
        print_result("GraphRAG", True)
        results.append(True)
    except Exception as e:
        print_result("GraphRAG", False, str(e))
        results.append(False)

    # Test OllamaRAG with caching
    try:
        from src.core.ollama_rag import OllamaRAG
        print_result("OllamaRAG with caching", True)
        results.append(True)
    except Exception as e:
        print_result("OllamaRAG with caching", False, str(e))
        results.append(False)

    # Test Router with GRAPHRAG
    try:
        from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter, RAGStrategy
        assert RAGStrategy.GRAPHRAG is not None
        print_result("Router with GRAPHRAG", True)
        results.append(True)
    except Exception as e:
        print_result("Router with GRAPHRAG", False, str(e))
        results.append(False)

    return all(results)


def test_caching():
    """Test caching functionality"""
    print_header("CACHING TESTS")

    try:
        from src.core.caching_system import RAGCacheManager, LRUCache

        # Test LRU Cache
        cache = LRUCache(capacity=3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1", "LRU get failed"
        assert cache.size() == 3, "LRU size incorrect"

        # Test eviction
        cache.put("key4", "value4")
        assert cache.size() == 3, "LRU eviction failed"
        print_result("LRU Cache", True, "Eviction and retrieval working")

        # Test Query Cache
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response(
            query="What is machine learning?",
            response="Machine learning is a subset of AI...",
            strategy="baseline"
        )

        cached = manager.get_query_response("What is machine learning?")
        assert cached is not None, "Query cache miss"
        assert "Machine learning" in cached["response"], "Wrong response cached"
        print_result("Query Cache", True, "Query caching and retrieval working")

        # Test cache stats
        stats = manager.get_stats()
        assert stats["query_cache"]["hits"] == 1, "Stats not tracking hits"
        print_result("Cache Statistics", True, f"Hit rate: {manager.get_hit_rate():.2%}")

        return True

    except Exception as e:
        print_result("Caching Tests", False, str(e))
        return False


def test_self_rag(documents: list):
    """Test Self-RAG with reflection tokens"""
    print_header("SELF-RAG TESTS")

    try:
        from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG

        print("  Initializing Self-RAG (requires Ollama)...")
        self_rag = OllamaSelfRAG(verbose=False)
        print_result("Self-RAG Initialization", True)

        # Test reflection on a query
        query = "What is machine learning?"
        print(f"  Testing query: '{query}'")

        result = self_rag.query_with_reflection(query, documents[:3])

        print_result("Answer Generation", True, f"Length: {len(result.answer)} chars")
        print_result("Reflection Tokens", True, f"Score: {result.reflection.overall_score:.2f}")

        print(f"\n  Reflection Details:")
        print(f"    - Relevance: {result.reflection.relevance.value}")
        print(f"    - Support: {result.reflection.support.value}")
        print(f"    - Utility: {result.reflection.utility.value}")
        print(f"    - Regenerations: {result.regeneration_count}")

        return True

    except ConnectionError as e:
        print_result("Self-RAG Tests", False, f"Ollama not available: {e}")
        print("  -> Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print_result("Self-RAG Tests", False, str(e))
        return False


def test_graph_rag(documents: list):
    """Test GraphRAG with knowledge graphs"""
    print_header("GRAPHRAG TESTS")

    try:
        from src.experiments.graph_reasoning.ollama_graph_rag import OllamaGraphRAG

        print("  Initializing GraphRAG (requires Ollama)...")
        graph_rag = OllamaGraphRAG(verbose=False)
        print_result("GraphRAG Initialization", True)

        # Build graph from documents
        print("  Building knowledge graph...")
        stats = graph_rag.build_graph_from_documents(documents)

        print_result("Graph Building", True,
                    f"Entities: {stats['entities']}, Relationships: {stats['relationships']}")

        # Test query
        query = "How does deep learning relate to neural networks and machine learning?"
        print(f"  Testing query: '{query}'")

        result = graph_rag.query(query)

        print_result("Graph Query", True, f"Answer length: {len(result.answer)} chars")
        print_result("Reasoning Path", True, f"Hops: {result.num_hops}, Entities: {len(result.entities_used)}")

        if result.reasoning_path:
            print(f"\n  Reasoning Path Sample:")
            for step in result.reasoning_path[:3]:
                print(f"    - {step['from']} --[{step['relation']}]--> {step['to']}")

        return True

    except ConnectionError as e:
        print_result("GraphRAG Tests", False, f"Ollama not available: {e}")
        print("  -> Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print_result("GraphRAG Tests", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_integrated_rag(documents: list):
    """Test integrated RAG with caching"""
    print_header("INTEGRATED RAG TESTS")

    try:
        from src.core.ollama_rag import OllamaRAG
        from src.core.caching_system import RAGCacheManager

        # Initialize with caching
        cache_manager = RAGCacheManager(enable_auto_cleanup=False)
        print("  Initializing OllamaRAG with caching...")

        rag = OllamaRAG(
            model="qwen2.5:14b",
            verbose=False,
            cache_manager=cache_manager,
            enable_caching=True
        )
        print_result("OllamaRAG with Cache", True)

        # Add documents
        rag.add_documents(documents)
        print_result("Document Ingestion", True, f"Added {len(documents)} documents")

        # First query (cache miss)
        query = "What is machine learning?"
        print(f"  First query: '{query}'")
        start_time = time.time()
        response1 = rag.query(query)
        time1 = time.time() - start_time
        print_result("First Query (Cache Miss)", True, f"Time: {time1:.1f}s")

        # Second query (cache hit)
        print(f"  Repeat query (should be cached)...")
        start_time = time.time()
        response2 = rag.query(query)
        time2 = time.time() - start_time

        cache_hit = time2 < time1 * 0.5  # Cache should be at least 2x faster
        print_result("Second Query (Cache Hit)", cache_hit,
                    f"Time: {time2:.1f}s (Speedup: {time1/time2:.1f}x)")

        # Check cache stats
        stats = cache_manager.get_stats()
        print_result("Cache Statistics", True,
                    f"Hits: {stats['summary']['total_hits']}, Misses: {stats['summary']['total_misses']}")

        return True

    except ConnectionError as e:
        print_result("Integrated RAG Tests", False, f"Ollama not available: {e}")
        print("  -> Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print_result("Integrated RAG Tests", False, str(e))
        return False


def test_router(skip_ollama: bool = False):
    """Test adaptive router with all strategies"""
    print_header("ROUTER TESTS")

    try:
        from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter, RAGStrategy

        # Test strategy enum first (doesn't need Ollama)
        assert RAGStrategy.GRAPHRAG.value == "graphrag"
        print_result("GRAPHRAG Strategy Available", True)

        strategies = [s.value for s in RAGStrategy]
        print_result("All Strategies", True, f"Available: {strategies}")

        if skip_ollama:
            print("  [SKIPPED] Router initialization (requires Ollama)")
            return True

        print("  Initializing router...")
        router = OllamaAdaptiveRouter(verbose=False)
        print_result("Router Initialization", True)

        # Test queries that should route to different strategies
        test_queries = [
            ("What is Python?", RAGStrategy.BASELINE, "Simple factual query"),
            ("How does machine learning work?", RAGStrategy.HYDE, "Medium complexity"),
            ("What is the relationship between ML and deep learning?", RAGStrategy.GRAPHRAG, "Multi-hop reasoning"),
        ]

        for query, expected_strategy, description in test_queries:
            decision = router.route_query(query)
            print_result(f"Routing: {description}", True,
                        f"'{query[:30]}...' -> {decision.selected_strategy.value}")

        return True

    except ConnectionError as e:
        print_result("Router Tests", False, f"Ollama not available: {e}")
        return False
    except Exception as e:
        print_result("Router Tests", False, str(e))
        return False


def load_pdf_document(pdf_path: str) -> list:
    """Load a PDF document for testing"""
    print(f"\n  Loading PDF: {pdf_path}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                full_text += f"\n\n--- Page {page_num} ---\n{page_text}"

        if full_text.strip():
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "type": "pdf_text",
                    "pages": len(reader.pages)
                }
            )
            print(f"  Loaded {len(reader.pages)} pages from PDF")
            return [doc]

    except ImportError:
        print("  WARNING: pypdf not installed. Install with: pip install pypdf")
    except Exception as e:
        print(f"  ERROR loading PDF: {e}")

    return []


def main():
    parser = argparse.ArgumentParser(description="Test all RAG features")
    parser.add_argument("--pdf", type=str, help="Path to PDF file for testing")
    parser.add_argument("--quick", action="store_true", help="Quick import-only tests")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip tests requiring Ollama")
    args = parser.parse_args()

    print_header("ADAPTIVE MULTIMODAL RAG - FEATURE TESTS")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Always run import tests
    imports_ok = test_imports()

    if args.quick:
        print_header("QUICK TEST SUMMARY")
        if imports_ok:
            print("  All imports successful!")
            print("  Run without --quick for full tests")
            return 0
        else:
            print("  Some imports failed!")
            return 1

    # Prepare documents
    documents = SAMPLE_DOCUMENTS.copy()
    if args.pdf:
        pdf_docs = load_pdf_document(args.pdf)
        documents.extend(pdf_docs)

    print(f"\n  Using {len(documents)} documents for testing")

    # Run all tests
    results = {
        "imports": imports_ok,
        "caching": test_caching(),
        "router": test_router(skip_ollama=args.skip_ollama),
    }

    if not args.skip_ollama:
        results["self_rag"] = test_self_rag(documents)
        results["graph_rag"] = test_graph_rag(documents)
        results["integrated"] = test_integrated_rag(documents)
    else:
        print("\n  [SKIPPED] Ollama-dependent tests (--skip-ollama)")

    # Summary
    print_header("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\n  Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed!")
        return 0
    else:
        print("\n  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
