#!/usr/bin/env python3
"""
Integration quality test suite - runs real queries against real PDFs with Ollama.
Evaluates retrieval quality, routing accuracy, and answer quality.

Usage:
    python tests/integration/run_quality_tests.py
    python tests/integration/run_quality_tests.py --pdf data/datasets/1810.04805v2.pdf
    python tests/integration/run_quality_tests.py --all-pdfs
"""

import sys
import os
import json
import time
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

# Add project root
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from langchain.schema import Document

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ── Test Case Definitions ────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single test query with expected behavior."""
    query: str
    category: str  # factual, summarization, analytical, metadata, comparison
    expected_strategy: Optional[str] = None  # expected routing strategy
    must_contain: List[str] = field(default_factory=list)  # keywords that MUST appear in answer
    must_not_contain: List[str] = field(default_factory=list)  # keywords that should NOT appear
    min_answer_length: int = 50  # minimum acceptable answer length
    max_answer_length: int = 5000  # maximum before it's too verbose


@dataclass
class TestResult:
    """Result of running a single test case."""
    test_case: TestCase
    answer: str = ""
    routed_strategy: str = ""
    complexity_score: int = 0
    num_chunks_retrieved: int = 0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0

    # Quality scores (0-10)
    routing_correct: bool = True
    length_ok: bool = True
    contains_required: bool = True
    missing_keywords: List[str] = field(default_factory=list)
    contains_forbidden: bool = False
    forbidden_found: List[str] = field(default_factory=list)
    answer_is_error: bool = False
    answer_is_hallucination: bool = False  # answered without doc grounding

    # Verification data
    verification: Optional[Dict] = None

    @property
    def passed(self) -> bool:
        return (
            self.length_ok
            and self.contains_required
            and not self.contains_forbidden
            and not self.answer_is_error
            and self.routing_correct
        )

    @property
    def score(self) -> float:
        """Overall quality score 0-10."""
        s = 10.0
        if not self.length_ok:
            s -= 3.0
        if not self.contains_required:
            s -= 3.0
        if self.contains_forbidden:
            s -= 2.0
        if self.answer_is_error:
            s -= 5.0
        if not self.routing_correct:
            s -= 1.0
        if self.answer_is_hallucination:
            s -= 2.0
        return max(0.0, s)


# ── Paper-Specific Test Suites ───────────────────────────────────────────────

# BERT paper (1810.04805v2.pdf)
BERT_TESTS = [
    TestCase(
        query="What is the title of this paper?",
        category="metadata",
        must_contain=["BERT", "pre-training"],
        expected_strategy="baseline",
    ),
    TestCase(
        query="What are the two pre-training tasks used in BERT?",
        category="factual",
        must_contain=["masked", "next sentence"],
        expected_strategy="baseline",
    ),
    TestCase(
        query="Give me a summary of this paper",
        category="summarization",
        must_contain=["BERT", "language"],
        expected_strategy="hyde",
        min_answer_length=200,
    ),
    TestCase(
        query="How does BERT differ from GPT?",
        category="comparison",
        must_contain=["bidirectional"],
        expected_strategy="hyde",
    ),
    TestCase(
        query="What datasets were used to evaluate BERT?",
        category="factual",
        must_contain=["GLUE"],
    ),
    TestCase(
        query="Explain the masked language model objective",
        category="analytical",
        must_contain=["mask", "token"],
        expected_strategy="hyde",
    ),
    TestCase(
        query="What is the model architecture of BERT?",
        category="factual",
        must_contain=["transformer"],
    ),
    TestCase(
        query="What results did BERT achieve on the SQuAD benchmark?",
        category="factual",
        must_contain=["SQuAD"],
    ),
]

# Attention paper (1706.03762v7.pdf)
ATTENTION_TESTS = [
    TestCase(
        query="What is the title of this paper?",
        category="metadata",
        must_contain=["attention"],
    ),
    TestCase(
        query="Summarize this paper",
        category="summarization",
        must_contain=["attention", "transformer"],
        expected_strategy="hyde",
        min_answer_length=200,
    ),
    TestCase(
        query="What is multi-head attention?",
        category="factual",
        must_contain=["head", "attention"],
    ),
    TestCase(
        query="What is the computational complexity of self-attention?",
        category="analytical",
        must_contain=["complexity"],
    ),
    TestCase(
        query="What BLEU scores were achieved on translation tasks?",
        category="factual",
        must_contain=["BLEU"],
    ),
]

# ResNet paper (1512.03385v1.pdf)
RESNET_TESTS = [
    TestCase(
        query="What is the title of this paper?",
        category="metadata",
        must_contain=["deep", "learning"],
    ),
    TestCase(
        query="What problem do residual connections solve?",
        category="analytical",
        must_contain=["degradation"],
    ),
    TestCase(
        query="Summarize this paper",
        category="summarization",
        must_contain=["residual"],
        expected_strategy="hyde",
        min_answer_length=200,
    ),
]

# RAG paper (2005.11401v4.pdf)
RAG_TESTS = [
    TestCase(
        query="What is RAG?",
        category="factual",
        must_contain=["retrieval"],
    ),
    TestCase(
        query="Summarize this paper",
        category="summarization",
        must_contain=["retrieval", "generation"],
        expected_strategy="hyde",
        min_answer_length=200,
    ),
    TestCase(
        query="How does RAG compare to closed-book approaches?",
        category="comparison",
        must_contain=["retrieval"],
    ),
]

# Generic tests that work on any paper
GENERIC_TESTS = [
    TestCase(
        query="What is the main contribution of this paper?",
        category="summarization",
        min_answer_length=100,
        expected_strategy="hyde",
    ),
    TestCase(
        query="What methodology was used?",
        category="factual",
        min_answer_length=100,
    ),
    TestCase(
        query="What are the key findings?",
        category="summarization",
        min_answer_length=100,
        expected_strategy="hyde",
    ),
]

# Map filenames to test suites
PAPER_TESTS = {
    "1810.04805v2.pdf": BERT_TESTS,
    "1706.03762v7.pdf": ATTENTION_TESTS,
    "1512.03385v1.pdf": RESNET_TESTS,
    "2005.11401v4.pdf": RAG_TESTS,
}


# ── Test Runner ──────────────────────────────────────────────────────────────

class QualityTestRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.rag = None
        self.router = None
        self.hyde_rag = None
        self.results: List[TestResult] = []

    def initialize(self):
        """Initialize RAG components."""
        print("Initializing RAG system...")
        from src.core.ollama_rag import OllamaRAG
        # Use temp directory to avoid cross-contamination from persisted vector stores
        self._temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        self.rag = OllamaRAG(verbose=False, enable_caching=False, persist_directory=self._temp_dir)
        print(f"  Model: {self.rag.model}")

        try:
            from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer
            from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter
            analyzer = OllamaQueryAnalyzer(model=self.rag.model, verbose=False)
            self.router = OllamaAdaptiveRouter(query_analyzer=analyzer, verbose=False)
            print("  Router: OK")
        except Exception as e:
            print(f"  Router: FAILED ({e})")

        try:
            from src.experiments.hyde.ollama_hyde import OllamaHyDE
            self.hyde_rag = OllamaHyDE(model=self.rag.model, verbose=False)
            print("  HyDE: OK")
        except Exception as e:
            print(f"  HyDE: FAILED ({e})")

    def load_pdf(self, pdf_path: str) -> int:
        """Load a PDF and return number of pages."""
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        docs = []
        filename = os.path.basename(pdf_path)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=f"[Page {page_num}] {text}",
                    metadata={
                        "source": filename,
                        "type": "pdf_text",
                        "page": page_num,
                        "total_pages": len(reader.pages),
                    },
                )
                docs.append(doc)

        # Reset vector store for clean test
        self.rag.vector_store = None
        self.rag.documents = []
        self.rag.add_documents(docs)
        print(f"  Loaded {len(docs)} pages from {filename}")
        return len(docs)

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case and evaluate."""
        result = TestResult(test_case=test)
        t_start = time.time()

        # Step 1: Route
        if self.router:
            try:
                decision = self.router.route_query(test.query)
                result.routed_strategy = decision.selected_strategy.value
                result.complexity_score = decision.complexity_score
            except Exception as e:
                result.routed_strategy = "error"
                print(f"    Routing error: {e}")

        # Step 2: Retrieve chunks
        t_retrieve = time.time()
        try:
            chunks = self.rag.retrieve_documents(test.query, k=10)
            result.num_chunks_retrieved = len(chunks)
        except Exception:
            result.num_chunks_retrieved = 0
        result.retrieval_time = time.time() - t_retrieve

        # Step 3: Generate answer
        t_gen = time.time()
        try:
            result.answer = self.rag.query(test.query, bypass_cache=True)
        except Exception as e:
            result.answer = f"ERROR: {e}"
            result.answer_is_error = True
        result.generation_time = time.time() - t_gen
        result.total_time = time.time() - t_start

        # Step 4: Evaluate
        self._evaluate(result)
        return result

    def run_verification_test(self, test: TestCase) -> TestResult:
        """Run a test with verification mode for groundedness checking."""
        result = self.run_test(test)

        try:
            verification = self.rag.query_with_verification(test.query)
            result.verification = {
                "rag_answer_len": len(verification.get("answer_with_retrieval", "")),
                "llm_answer_len": len(verification.get("answer_without_retrieval", "")),
                "notes": verification.get("verification_notes", ""),
                "num_docs": len(verification.get("retrieved_docs", [])),
            }

            # Check for hallucination: if RAG and LLM answers are very similar,
            # the model may be using training data not documents
            rag_ans = verification.get("answer_with_retrieval", "").lower()
            llm_ans = verification.get("answer_without_retrieval", "").lower()
            if rag_ans and llm_ans:
                rag_words = set(rag_ans.split())
                llm_words = set(llm_ans.split())
                if rag_words and llm_words:
                    overlap = len(rag_words & llm_words) / len(rag_words | llm_words)
                    result.verification["word_overlap"] = round(overlap, 3)
                    # Very high overlap suggests LLM is not using docs
                    if overlap > 0.85:
                        result.answer_is_hallucination = True
        except Exception as e:
            result.verification = {"error": str(e)}

        return result

    def _evaluate(self, result: TestResult):
        """Evaluate a test result."""
        answer_lower = result.answer.lower()

        # Check for error
        if result.answer.startswith("ERROR") or result.answer.startswith("Error generating"):
            result.answer_is_error = True

        # Check length
        result.length_ok = (
            result.test_case.min_answer_length <= len(result.answer) <= result.test_case.max_answer_length
        )

        # Check required keywords
        result.missing_keywords = [
            kw for kw in result.test_case.must_contain
            if kw.lower() not in answer_lower
        ]
        result.contains_required = len(result.missing_keywords) == 0

        # Check forbidden keywords
        result.forbidden_found = [
            kw for kw in result.test_case.must_not_contain
            if kw.lower() in answer_lower
        ]
        result.contains_forbidden = len(result.forbidden_found) > 0

        # Check routing
        if result.test_case.expected_strategy and result.routed_strategy:
            result.routing_correct = (
                result.routed_strategy == result.test_case.expected_strategy
            )
        else:
            result.routing_correct = True  # no expectation set

    def run_suite(self, tests: List[TestCase], verify_sample: bool = True) -> List[TestResult]:
        """Run a full test suite."""
        results = []
        total = len(tests)

        for i, test in enumerate(tests, 1):
            print(f"\n  [{i}/{total}] {test.category}: \"{test.query[:60]}...\"" if len(test.query) > 60 else f"\n  [{i}/{total}] {test.category}: \"{test.query}\"")

            # Run with verification for first 2 tests, or all summarization tests
            if verify_sample and (i <= 2 or test.category == "summarization"):
                result = self.run_verification_test(test)
            else:
                result = self.run_test(test)

            results.append(result)
            self.results.append(result)

            # Print compact result
            status = "PASS" if result.passed else "FAIL"
            strategy_info = f"strategy={result.routed_strategy}" if result.routed_strategy else ""
            print(f"    {status} | score={result.score:.1f}/10 | {strategy_info} | chunks={result.num_chunks_retrieved} | time={result.total_time:.1f}s | len={len(result.answer)}")

            if not result.passed:
                if result.answer_is_error:
                    print(f"    ERROR: {result.answer[:200]}")
                if result.missing_keywords:
                    print(f"    MISSING: {result.missing_keywords}")
                if not result.length_ok:
                    print(f"    LENGTH: {len(result.answer)} (expected {result.test_case.min_answer_length}-{result.test_case.max_answer_length})")
                if not result.routing_correct:
                    print(f"    ROUTING: got {result.routed_strategy}, expected {result.test_case.expected_strategy}")
                if result.answer_is_hallucination:
                    print(f"    HALLUCINATION: high overlap with LLM-only answer")

            if result.verification:
                overlap = result.verification.get("word_overlap", None)
                if overlap is not None:
                    grounding = "GOOD" if overlap < 0.5 else "WEAK" if overlap < 0.85 else "POOR"
                    print(f"    GROUNDING: {grounding} (overlap={overlap:.1%})")

        return results

    def print_report(self, results: Optional[List[TestResult]] = None):
        """Print a summary report."""
        results = results or self.results
        if not results:
            print("\nNo results to report.")
            return

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / total
        avg_time = sum(r.total_time for r in results) / total

        print(f"\n{'=' * 70}")
        print(f"  QUALITY TEST REPORT")
        print(f"{'=' * 70}")
        print(f"  Tests: {passed}/{total} passed ({passed/total:.0%})")
        print(f"  Average score: {avg_score:.1f}/10")
        print(f"  Average time: {avg_time:.1f}s per query")

        # By category
        categories = set(r.test_case.category for r in results)
        print(f"\n  By Category:")
        for cat in sorted(categories):
            cat_results = [r for r in results if r.test_case.category == cat]
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_score = sum(r.score for r in cat_results) / len(cat_results)
            print(f"    {cat:15s}: {cat_passed}/{len(cat_results)} passed, avg score {cat_score:.1f}/10")

        # Routing accuracy
        routed = [r for r in results if r.test_case.expected_strategy]
        if routed:
            route_correct = sum(1 for r in routed if r.routing_correct)
            print(f"\n  Routing accuracy: {route_correct}/{len(routed)} ({route_correct/len(routed):.0%})")

        # Failures detail
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n  Failures ({len(failures)}):")
            for r in failures:
                issues = []
                if r.answer_is_error:
                    issues.append("ERROR")
                if r.missing_keywords:
                    issues.append(f"missing={r.missing_keywords}")
                if not r.length_ok:
                    issues.append(f"len={len(r.answer)}")
                if not r.routing_correct:
                    issues.append(f"route={r.routed_strategy}!={r.test_case.expected_strategy}")
                print(f"    - \"{r.test_case.query[:50]}\" [{', '.join(issues)}]")

        # Grounding check
        verified = [r for r in results if r.verification and "word_overlap" in r.verification]
        if verified:
            avg_overlap = sum(r.verification["word_overlap"] for r in verified) / len(verified)
            hallucinated = sum(1 for r in verified if r.answer_is_hallucination)
            print(f"\n  Grounding: avg overlap={avg_overlap:.1%}, potential hallucinations={hallucinated}/{len(verified)}")

        print(f"{'=' * 70}")

        return {
            "total": total,
            "passed": passed,
            "avg_score": round(avg_score, 1),
            "avg_time": round(avg_time, 1),
            "failures": [
                {
                    "query": r.test_case.query,
                    "category": r.test_case.category,
                    "score": r.score,
                    "issues": {
                        "error": r.answer_is_error,
                        "missing_keywords": r.missing_keywords,
                        "length_ok": r.length_ok,
                        "routing_correct": r.routing_correct,
                        "hallucination": r.answer_is_hallucination,
                    },
                    "answer_preview": r.answer[:300],
                    "routed_strategy": r.routed_strategy,
                }
                for r in results if not r.passed
            ],
        }

    def save_results(self, filepath: str):
        """Save detailed results to JSON."""
        data = []
        for r in self.results:
            entry = {
                "query": r.test_case.query,
                "category": r.test_case.category,
                "expected_strategy": r.test_case.expected_strategy,
                "must_contain": r.test_case.must_contain,
                "answer": r.answer,
                "answer_length": len(r.answer),
                "routed_strategy": r.routed_strategy,
                "complexity_score": r.complexity_score,
                "num_chunks": r.num_chunks_retrieved,
                "total_time": round(r.total_time, 2),
                "retrieval_time": round(r.retrieval_time, 2),
                "generation_time": round(r.generation_time, 2),
                "passed": r.passed,
                "score": r.score,
                "routing_correct": r.routing_correct,
                "length_ok": r.length_ok,
                "missing_keywords": r.missing_keywords,
                "answer_is_error": r.answer_is_error,
                "answer_is_hallucination": r.answer_is_hallucination,
                "verification": r.verification,
            }
            data.append(entry)

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nDetailed results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="RAG Quality Test Runner")
    parser.add_argument("--pdf", help="Specific PDF to test")
    parser.add_argument("--all-pdfs", action="store_true", help="Test all known papers")
    parser.add_argument("--generic-only", action="store_true", help="Only run generic tests")
    parser.add_argument("--output", default="tests/integration/results.json", help="Output JSON path")
    args = parser.parse_args()

    runner = QualityTestRunner()
    runner.initialize()

    data_dir = os.path.join(PROJECT_ROOT, "data", "datasets")

    if args.pdf:
        pdf_path = args.pdf if os.path.isabs(args.pdf) else os.path.join(PROJECT_ROOT, args.pdf)
        filename = os.path.basename(pdf_path)
        print(f"\nLoading {filename}...")
        runner.load_pdf(pdf_path)

        tests = PAPER_TESTS.get(filename, []) + GENERIC_TESTS
        if args.generic_only:
            tests = GENERIC_TESTS
        print(f"\nRunning {len(tests)} tests on {filename}...")
        runner.run_suite(tests)

    elif args.all_pdfs:
        for filename, tests in PAPER_TESTS.items():
            pdf_path = os.path.join(data_dir, filename)
            if not os.path.exists(pdf_path):
                print(f"\nSkipping {filename} (not found)")
                continue
            print(f"\n{'#' * 60}")
            print(f"  Testing: {filename}")
            print(f"{'#' * 60}")
            runner.load_pdf(pdf_path)
            all_tests = tests + GENERIC_TESTS
            runner.run_suite(all_tests)
    else:
        # Default: test BERT paper
        bert_path = os.path.join(data_dir, "1810.04805v2.pdf")
        if os.path.exists(bert_path):
            print("\nDefault: testing BERT paper...")
            runner.load_pdf(bert_path)
            runner.run_suite(BERT_TESTS + GENERIC_TESTS)
        else:
            print(f"BERT paper not found at {bert_path}")
            return

    report = runner.print_report()
    runner.save_results(os.path.join(PROJECT_ROOT, args.output))

    # Return exit code based on pass rate
    if report and report["passed"] < report["total"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
