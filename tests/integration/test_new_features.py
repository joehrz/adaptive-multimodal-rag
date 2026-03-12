"""
Integration test for semantic chunking and evaluation benchmarks.
Requires Ollama running with qwen2.5:14b.
"""

import sys
import os
import tempfile
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pypdf import PdfReader
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "datasets")
BERT_PDF = os.path.join(DATASETS, "1810.04805v2.pdf")
ATTENTION_PDF = os.path.join(DATASETS, "1706.03762v7.pdf")


def load_pdf(path):
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                page_content=f"[Page {i+1}] {text}",
                metadata={"source": os.path.basename(path), "page": i+1, "type": "pdf_text"}
            ))
    return docs


def test_semantic_chunking():
    """Test that semantic chunking produces reasonable chunks."""
    print("\n" + "="*70)
    print("TEST 1: SEMANTIC CHUNKING")
    print("="*70)

    from src.core.chunking import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    chunker = SemanticChunker(
        embeddings=embeddings,
        min_chunk_size=200,
        max_chunk_size=2000,
        similarity_threshold_percentile=25,
    )

    # Load first 3 pages of BERT paper
    docs = load_pdf(BERT_PDF)[:3]
    print(f"Loaded {len(docs)} pages from BERT paper")

    start = time.time()
    chunks = chunker.split_documents(docs)
    elapsed = time.time() - start

    print(f"Semantic chunking produced {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"Chunk sizes: {[len(c.page_content) for c in chunks]}")
    print(f"Min chunk size: {min(len(c.page_content) for c in chunks)}")
    print(f"Max chunk size: {max(len(c.page_content) for c in chunks)}")
    print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f}")

    # Compare with recursive chunking
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    recursive_chunks = recursive.split_documents(docs)
    print(f"\nRecursive chunking produced {len(recursive_chunks)} chunks")
    print(f"Recursive avg chunk size: {sum(len(c.page_content) for c in recursive_chunks) / len(recursive_chunks):.0f}")

    # Verify metadata preserved
    for chunk in chunks:
        assert "source" in chunk.metadata, "Missing source metadata"
        assert "chunk_index" in chunk.metadata, "Missing chunk_index metadata"

    print("\nSemantic chunking: PASSED")
    return True


def test_semantic_chunking_rag():
    """Test semantic chunking integrated into the RAG pipeline."""
    print("\n" + "="*70)
    print("TEST 2: SEMANTIC CHUNKING IN RAG PIPELINE")
    print("="*70)

    from src.core.ollama_rag import OllamaRAG

    tmpdir = tempfile.mkdtemp()
    rag = OllamaRAG(
        model="qwen2.5:14b",
        persist_directory=tmpdir,
        chunk_size=1000,
        chunk_overlap=200,
        verbose=True,
    )

    # Force semantic chunking
    from src.core.chunking import SemanticChunker
    rag.text_splitter = SemanticChunker(
        embeddings=rag.embeddings,
        min_chunk_size=200,
        max_chunk_size=2000,
    )

    docs = load_pdf(BERT_PDF)
    print(f"Loading {len(docs)} pages into RAG with semantic chunking...")
    rag.add_documents(docs)

    queries = [
        "What is BERT?",
        "What are the two pre-training objectives used by BERT?",
        "What results did BERT achieve on the GLUE benchmark?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        answer = rag.query(query)
        print(f"Answer: {answer[:200]}...")
        assert len(answer) > 50, f"Answer too short: {len(answer)} chars"
        assert "error" not in answer.lower() or "error" in query.lower(), "Got error response"

    print("\nSemantic chunking RAG: PASSED")
    return True


def test_evaluation_metrics():
    """Test the evaluation metrics module."""
    print("\n" + "="*70)
    print("TEST 3: EVALUATION METRICS")
    print("="*70)

    from src.evaluation.metrics import RAGEvaluator, compute_rouge_l

    # Test ROUGE-L first (no LLM needed)
    rouge = compute_rouge_l(
        "BERT uses masked language modeling and next sentence prediction",
        "BERT is pre-trained with masked language modeling and next sentence prediction objectives"
    )
    print(f"ROUGE-L: precision={rouge.precision:.3f}, recall={rouge.recall:.3f}, f1={rouge.f1:.3f}")
    assert rouge.f1 > 0.3, f"ROUGE-L F1 too low: {rouge.f1}"
    print("ROUGE-L: PASSED")

    # Test LLM-as-judge metrics
    evaluator = RAGEvaluator(model="qwen2.5:14b")

    context = (
        "BERT stands for Bidirectional Encoder Representations from Transformers. "
        "It is pre-trained using two objectives: masked language modeling (MLM) and "
        "next sentence prediction (NSP). BERT achieved state-of-the-art results on "
        "11 NLP benchmarks including GLUE, SQuAD, and MultiNLI."
    )
    answer = (
        "BERT uses two pre-training objectives: masked language modeling, where "
        "random tokens are masked and predicted from context, and next sentence "
        "prediction, where the model determines if two sentences are consecutive."
    )
    question = "What are BERT's pre-training objectives?"

    print("\nEvaluating faithfulness...")
    faith = evaluator.evaluate_faithfulness(context, answer)
    print(f"  Faithfulness: {faith.score:.3f} (error: {faith.error})")

    print("Evaluating answer relevance...")
    rel = evaluator.evaluate_answer_relevance(question, answer)
    print(f"  Answer Relevance: {rel.score:.3f} (error: {rel.error})")

    print("Evaluating context precision...")
    prec = evaluator.evaluate_context_precision(question, [context])
    print(f"  Context Precision: {prec.score:.3f} (error: {prec.error})")

    # All scores should be reasonably high for this well-grounded answer
    assert faith.score >= 0.5 or faith.error, f"Faithfulness too low: {faith.score}"
    assert rel.score >= 0.5 or rel.error, f"Relevance too low: {rel.score}"

    print("\nEvaluation metrics: PASSED")
    return True


def test_benchmark_runner():
    """Test the benchmark runner against the Attention paper."""
    print("\n" + "="*70)
    print("TEST 4: BENCHMARK RUNNER (Attention paper)")
    print("="*70)

    from src.core.ollama_rag import OllamaRAG
    from src.evaluation.benchmark import BenchmarkRunner, ATTENTION_BENCHMARK

    tmpdir = tempfile.mkdtemp()
    rag = OllamaRAG(
        model="qwen2.5:14b",
        persist_directory=tmpdir,
        verbose=False,
    )

    docs = load_pdf(ATTENTION_PDF)
    print(f"Loading {len(docs)} pages from Attention paper...")
    rag.add_documents(docs)

    # Run benchmark on just 3 cases to keep it fast
    runner = BenchmarkRunner(evaluator_model="qwen2.5:14b")
    cases = ATTENTION_BENCHMARK[:3]
    print(f"Running {len(cases)} benchmark cases...")

    report = runner.run(rag, cases, name="Attention paper test")
    print(f"\n{report.summary()}")

    scores = report.aggregate_scores()
    print(f"\nAggregate score: {scores['aggregate']:.3f}")
    print(f"ROUGE-L F1: {scores['rouge_l_f1']:.3f}")

    assert scores["aggregate"] > 0, "All metrics returned 0"
    assert report.num_cases == 3, f"Expected 3 cases, got {report.num_cases}"

    # Save results
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_test_results.json")
    report.save(outpath)
    print(f"Results saved to {outpath}")

    print("\nBenchmark runner: PASSED")
    return True


if __name__ == "__main__":
    results = {}

    for test_fn in [test_semantic_chunking, test_semantic_chunking_rag, test_evaluation_metrics, test_benchmark_runner]:
        name = test_fn.__name__
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n{name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{total} tests passed")
