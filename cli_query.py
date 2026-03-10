#!/usr/bin/env python3
"""
CLI tool for testing RAG without Streamlit.

Usage:
    python cli_query.py                              # Interactive mode with file picker
    python cli_query.py --pdf data/datasets/1810.04805v2.pdf  # Load specific PDF
    python cli_query.py --pdf data/datasets/*.pdf    # Load multiple PDFs
    python cli_query.py --quick                      # Quick test with sample docs

Commands (in interactive mode):
    Type a question to query
    /verify <question>   - Run with groundedness verification
    /chunks <question>   - Show retrieved chunks only
    /route <question>    - Show routing decision without querying
    /strategy <name>     - Force a strategy (baseline/hyde/self_rag/graphrag)
    /stats               - Show routing stats
    /reload              - Re-list and load PDFs
    /quit                - Exit
"""

import sys
import os
import argparse
import time
import glob
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── PDF Loading ──────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF file and return page-level Documents."""
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
    return docs


def list_available_pdfs(data_dir: str = "data/datasets") -> List[str]:
    """List all PDFs in the datasets directory."""
    pattern = os.path.join(data_dir, "*.pdf")
    pdfs = sorted(glob.glob(pattern))
    return pdfs


def pick_pdf_interactive(data_dir: str = "data/datasets") -> List[str]:
    """Let user pick PDFs interactively."""
    pdfs = list_available_pdfs(data_dir)
    if not pdfs:
        print(f"No PDFs found in {data_dir}/")
        return []

    print(f"\nAvailable PDFs in {data_dir}/:")
    print("-" * 60)
    for i, pdf in enumerate(pdfs, 1):
        name = os.path.basename(pdf)
        size_mb = os.path.getsize(pdf) / (1024 * 1024)
        print(f"  {i:2d}. {name} ({size_mb:.1f} MB)")
    print(f"  {'a':>2s}. Load all")
    print()

    choice = input("Select PDF(s) (comma-separated numbers, or 'a' for all): ").strip()
    if not choice:
        return []

    if choice.lower() == "a":
        return pdfs

    selected = []
    for part in choice.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(pdfs):
                selected.append(pdfs[idx])
            else:
                print(f"  Skipping invalid index: {part}")
        except ValueError:
            print(f"  Skipping invalid input: {part}")
    return selected


# ── Sample Documents ─────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    Document(
        page_content="Machine learning is a subset of AI that enables computers to learn from data. Deep learning uses neural networks with multiple layers.",
        metadata={"source": "sample", "type": "sample", "page": 1},
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for more accurate responses.",
        metadata={"source": "sample", "type": "sample", "page": 2},
    ),
    Document(
        page_content="Transformers use attention mechanisms for parallel processing. BERT and GPT are transformer-based models.",
        metadata={"source": "sample", "type": "sample", "page": 3},
    ),
]


# ── Display Helpers ──────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(title: str, content: str):
    print(f"\n--- {title} ---")
    print(content)


def print_chunks(docs: List[Document], max_chars: int = 300):
    """Display retrieved chunks."""
    if not docs:
        print("  (no chunks retrieved)")
        return
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:max_chars].replace("\n", " ")
        if len(doc.page_content) > max_chars:
            preview += "..."
        print(f"\n  [{i}] Source: {source} | Page: {page}")
        print(f"      {preview}")


def print_routing(decision):
    """Display routing decision details."""
    print(f"\n  Strategy:   {decision.selected_strategy.value}")
    print(f"  Complexity: {decision.complexity_score}/10 ({decision.complexity_level.value})")
    print(f"  Reasoning:  {decision.reasoning}")
    print(f"  Exp. Quality: {decision.expected_quality:.1f} | Exp. Latency: {decision.expected_latency:.1f}s")
    print(f"  Route time: {decision.routing_time:.3f}s")


def print_verification(result: dict):
    """Display verification results."""
    print_section("RAG Answer (with documents)", result["answer_with_retrieval"])
    print_section("LLM-Only Answer (no documents)", result["answer_without_retrieval"])

    notes = result.get("verification_notes", "")
    if notes:
        print_section("Verification", notes)

    docs = result.get("retrieved_docs", [])
    if docs:
        print(f"\n--- Retrieved Chunks ({len(docs)}) ---")
        for i, d in enumerate(docs, 1):
            snippet = d.get("content", d.get("page_content", ""))[:200]
            source = d.get("source", "?")
            page = d.get("page", "?")
            print(f"  [{i}] {source} p.{page}: {snippet}...")


# ── Main Engine ──────────────────────────────────────────────────────────────

class CLIQueryEngine:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.rag = None
        self.router = None
        self.hyde_rag = None
        self.self_rag = None
        self.graph_rag = None
        self.documents = []
        self.forced_strategy = None

    def initialize(self):
        """Initialize the RAG system and router."""
        print("Initializing RAG system...")
        from src.core.ollama_rag import OllamaRAG

        self.rag = OllamaRAG(verbose=self.verbose, enable_caching=False)
        print(f"  Model: {self.rag.model}")

        # Initialize router
        try:
            from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer
            from src.experiments.adaptive_routing.ollama_router import OllamaAdaptiveRouter

            analyzer = OllamaQueryAnalyzer(model=self.rag.model, verbose=False)
            self.router = OllamaAdaptiveRouter(query_analyzer=analyzer, verbose=False)
            print("  Router: initialized")
        except Exception as e:
            print(f"  Router: failed ({e})")

        # Initialize HyDE
        try:
            from src.experiments.hyde.ollama_hyde import OllamaHyDE
            self.hyde_rag = OllamaHyDE(model=self.rag.model, verbose=False)
            print("  HyDE: initialized")
        except Exception as e:
            print(f"  HyDE: failed ({e})")

        # Initialize Self-RAG
        try:
            from src.experiments.self_reflection.ollama_self_rag import OllamaSelfRAG
            self.self_rag = OllamaSelfRAG(model=self.rag.model, verbose=False)
            print("  Self-RAG: initialized")
        except Exception as e:
            print(f"  Self-RAG: failed ({e})")

        # Initialize GraphRAG
        try:
            from src.experiments.graph_reasoning.ollama_graph_rag import OllamaGraphRAG
            self.graph_rag = OllamaGraphRAG(model=self.rag.model, verbose=False)
            print("  GraphRAG: initialized")
        except Exception as e:
            print(f"  GraphRAG: failed ({e})")

    def load_documents(self, docs: List[Document]):
        """Load documents into the RAG system."""
        if not docs:
            print("No documents to load.")
            return
        self.documents = docs
        t0 = time.time()
        self.rag.add_documents(docs)
        elapsed = time.time() - t0
        print(f"Loaded {len(docs)} document chunks in {elapsed:.1f}s")

    def load_pdfs(self, pdf_paths: List[str]):
        """Load one or more PDFs."""
        all_docs = []
        for path in pdf_paths:
            print(f"  Loading {os.path.basename(path)}...")
            docs = load_pdf(path)
            print(f"    -> {len(docs)} pages extracted")
            all_docs.extend(docs)
        self.load_documents(all_docs)

    def route(self, query: str):
        """Route a query and return the decision."""
        if not self.router:
            print("Router not available, defaulting to baseline.")
            return None
        return self.router.route_query(query)

    def query(self, question: str) -> str:
        """Run a query through the full adaptive pipeline."""
        from src.experiments.adaptive_routing.ollama_router import RAGStrategy

        # Route query
        strategy = None
        if self.forced_strategy:
            strategy = self.forced_strategy
            print(f"\n  [Forced strategy: {strategy.value}]")
        elif self.router:
            decision = self.route(question)
            if decision:
                strategy = decision.selected_strategy
                print_routing(decision)

        if strategy is None:
            strategy = RAGStrategy.BASELINE

        t0 = time.time()
        answer = None

        if strategy == RAGStrategy.BASELINE:
            answer = self.rag.query(question, bypass_cache=True)

        elif strategy == RAGStrategy.HYDE and self.hyde_rag:
            result = self.hyde_rag.query(question)
            answer = result if isinstance(result, str) else result.get("answer", str(result))

        elif strategy == RAGStrategy.SELF_RAG and self.self_rag:
            docs = self.rag.retrieve_documents(question, k=5)
            result = self.self_rag.query_with_reflection(question, docs)
            answer = result if isinstance(result, str) else result.get("answer", str(result))

        elif strategy == RAGStrategy.HYDE_SELF_RAG and self.hyde_rag and self.self_rag:
            hyde_result = self.hyde_rag.query(question)
            hyde_answer = hyde_result if isinstance(hyde_result, str) else hyde_result.get("answer", str(hyde_result))
            docs = self.rag.retrieve_documents(question, k=5)
            result = self.self_rag.query_with_reflection(question, docs)
            answer = result if isinstance(result, str) else result.get("answer", str(result))

        elif strategy == RAGStrategy.GRAPHRAG and self.graph_rag:
            if self.graph_rag.graph.number_of_nodes() == 0 and self.documents:
                print("  Building knowledge graph...")
                self.graph_rag.build_graph_from_documents(self.documents)
            docs = self.rag.retrieve_documents(question, k=5)
            result = self.graph_rag.query(question, retrieved_docs=docs)
            answer = result if isinstance(result, str) else result.get("answer", str(result))

        else:
            # Fallback to baseline
            answer = self.rag.query(question, bypass_cache=True)

        elapsed = time.time() - t0
        print(f"\n  [{strategy.value}] answered in {elapsed:.1f}s")
        return answer

    def show_chunks(self, question: str):
        """Show retrieved chunks for a query."""
        docs = self.rag.retrieve_documents(question, k=10)
        print(f"\nRetrieved {len(docs)} chunks for: \"{question}\"")
        print_chunks(docs)

    def verify(self, question: str):
        """Run verification mode."""
        result = self.rag.query_with_verification(question)
        print_verification(result)

    def show_stats(self):
        """Show routing statistics."""
        if not self.router:
            print("Router not available.")
            return
        stats = self.router.get_stats()
        print(f"\nTotal queries routed: {stats['total_queries_routed']}")
        for strategy_name, s in stats.get("strategy_stats", {}).items():
            if s.get("count", 0) > 0:
                print(f"  {strategy_name}: {s['count']} queries, avg latency {s.get('avg_latency', 0):.1f}s")


def interactive_loop(engine: CLIQueryEngine):
    """Run the interactive query loop."""
    from src.experiments.adaptive_routing.ollama_router import RAGStrategy

    print_header("Interactive RAG Query Mode")
    print("Commands: /verify, /chunks, /route, /strategy, /stats, /reload, /quit")
    print("Type a question to query with adaptive routing.\n")

    while True:
        try:
            user_input = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Exiting.")
            break

        elif user_input.lower() == "/stats":
            engine.show_stats()

        elif user_input.lower() == "/reload":
            selected = pick_pdf_interactive()
            if selected:
                engine.load_pdfs(selected)

        elif user_input.lower().startswith("/strategy"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                current = engine.forced_strategy.value if engine.forced_strategy else "auto"
                strategies = [s.value for s in RAGStrategy]
                print(f"  Current: {current}")
                print(f"  Available: auto, {', '.join(strategies)}")
            else:
                name = parts[1].strip().lower()
                if name == "auto":
                    engine.forced_strategy = None
                    print("  Strategy: auto (adaptive routing)")
                else:
                    try:
                        engine.forced_strategy = RAGStrategy(name)
                        print(f"  Strategy forced to: {name}")
                    except ValueError:
                        print(f"  Unknown strategy: {name}")

        elif user_input.startswith("/verify "):
            question = user_input[8:].strip()
            if question:
                engine.verify(question)

        elif user_input.startswith("/chunks "):
            question = user_input[8:].strip()
            if question:
                engine.show_chunks(question)

        elif user_input.startswith("/route "):
            question = user_input[7:].strip()
            if question:
                decision = engine.route(question)
                if decision:
                    print_routing(decision)

        else:
            # Regular query
            answer = engine.query(user_input)
            print_section("Answer", answer)


def main():
    parser = argparse.ArgumentParser(description="CLI RAG Query Tool")
    parser.add_argument("--pdf", nargs="+", help="PDF file(s) to load")
    parser.add_argument("--quick", action="store_true", help="Quick test with sample documents")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--data-dir", default="data/datasets", help="Directory containing PDFs")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    engine = CLIQueryEngine(verbose=not args.quiet)
    engine.initialize()

    if args.quick:
        engine.load_documents(SAMPLE_DOCUMENTS)
    elif args.pdf:
        # Expand globs
        pdf_paths = []
        for pattern in args.pdf:
            expanded = glob.glob(pattern)
            pdf_paths.extend(expanded if expanded else [pattern])
        engine.load_pdfs(pdf_paths)
    else:
        # Interactive PDF selection
        selected = pick_pdf_interactive(args.data_dir)
        if selected:
            engine.load_pdfs(selected)
        else:
            print("No PDFs selected. Loading sample documents.")
            engine.load_documents(SAMPLE_DOCUMENTS)

    interactive_loop(engine)


if __name__ == "__main__":
    main()
