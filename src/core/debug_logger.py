"""
Debug Logger for Adaptive RAG System
Captures and logs all RAG interactions for debugging and analysis
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Captured retrieved document info"""
    rank: int
    source: str
    content_preview: str  # First N characters
    content_length: int
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugEntry:
    """Single debug log entry for one query"""
    # Metadata
    timestamp: str
    session_id: str
    entry_id: int

    # Input
    user_query: str

    # Query Analysis
    complexity_score: Optional[int] = None
    complexity_level: Optional[str] = None
    selected_strategy: Optional[str] = None
    routing_reasoning: Optional[str] = None

    # Retrieval
    total_docs_retrieved: int = 0
    unique_docs_retrieved: int = 0
    duplicate_docs_count: int = 0
    retrieved_documents: List[Dict] = field(default_factory=list)

    # HyDE (if used)
    hyde_enabled: bool = False
    hypothetical_document: Optional[str] = None
    hyde_generation_time: Optional[float] = None

    # Self-RAG Reflection (if used)
    self_rag_enabled: bool = False
    relevance_score: Optional[float] = None
    relevance_token: Optional[str] = None
    support_score: Optional[float] = None
    support_token: Optional[str] = None
    utility_score: Optional[float] = None
    utility_token: Optional[str] = None
    overall_reflection_score: Optional[float] = None
    regeneration_count: int = 0

    # GraphRAG (if used)
    graphrag_enabled: bool = False
    entities_used: List[str] = field(default_factory=list)
    relationships_used: List[str] = field(default_factory=list)
    num_hops: int = 0

    # LLM Output
    llm_response: Optional[str] = None
    response_length: int = 0

    # Timing
    query_analysis_time: Optional[float] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    reflection_time: Optional[float] = None
    total_time: Optional[float] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DebugLogger:
    """
    Debug logger for RAG system interactions

    Usage:
        debug_logger = DebugLogger(output_dir="./debug_logs")
        debug_logger.start_entry("What is BERT?")
        debug_logger.log_query_analysis(score=4, level="medium", strategy="hyde")
        debug_logger.log_retrieval(documents=[...])
        debug_logger.log_response("BERT is...")
        debug_logger.save_entry()
    """

    def __init__(
        self,
        output_dir: str = "./debug_logs",
        session_id: Optional[str] = None,
        enabled: bool = True,
        content_preview_length: int = 500,
        save_format: str = "both"  # "json", "txt", or "both"
    ):
        """
        Initialize debug logger

        Args:
            output_dir: Directory to save debug logs
            session_id: Unique session identifier (auto-generated if None)
            enabled: Whether logging is enabled
            content_preview_length: Max characters to save for document previews
            save_format: Output format - "json", "txt", or "both"
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.content_preview_length = content_preview_length
        self.save_format = save_format

        # Generate session ID if not provided
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current entry being built
        self.current_entry: Optional[DebugEntry] = None
        self.entry_count = 0

        # Session log file paths
        self.json_log_path = self.output_dir / f"debug_session_{self.session_id}.json"
        self.txt_log_path = self.output_dir / f"debug_session_{self.session_id}.txt"

        # Initialize session files
        if self.enabled:
            self._init_session_files()

    def _init_session_files(self):
        """Initialize session log files with headers"""
        # Initialize JSON file with empty array
        if self.save_format in ["json", "both"]:
            with open(self.json_log_path, 'w') as f:
                json.dump({"session_id": self.session_id, "entries": []}, f, indent=2)

        # Initialize TXT file with header
        if self.save_format in ["txt", "both"]:
            with open(self.txt_log_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"DEBUG LOG - Session: {self.session_id}\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

    def start_entry(self, user_query: str) -> None:
        """Start a new debug entry for a query"""
        if not self.enabled:
            return

        self.entry_count += 1
        self.current_entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            entry_id=self.entry_count,
            user_query=user_query
        )

    def log_query_analysis(
        self,
        complexity_score: Optional[int] = None,
        complexity_level: Optional[str] = None,
        selected_strategy: Optional[str] = None,
        routing_reasoning: Optional[str] = None,
        analysis_time: Optional[float] = None
    ) -> None:
        """Log query analysis results"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.complexity_score = complexity_score
        self.current_entry.complexity_level = complexity_level
        self.current_entry.selected_strategy = selected_strategy
        self.current_entry.routing_reasoning = routing_reasoning
        self.current_entry.query_analysis_time = analysis_time

    def log_retrieval(
        self,
        documents: List[Any],
        retrieval_time: Optional[float] = None,
        scores: Optional[List[float]] = None
    ) -> None:
        """Log retrieved documents"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.retrieval_time = retrieval_time
        self.current_entry.total_docs_retrieved = len(documents)

        # Track unique documents and duplicates
        seen_content = set()
        unique_count = 0

        for i, doc in enumerate(documents):
            # Handle different document types
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            elif isinstance(doc, dict):
                content = doc.get('content', doc.get('page_content', str(doc)))
                metadata = doc.get('metadata', {})
            else:
                content = str(doc)
                metadata = {}

            # Check for duplicates
            content_hash = hash(content[:200])  # Hash first 200 chars for comparison
            is_duplicate = content_hash in seen_content
            seen_content.add(content_hash)

            if not is_duplicate:
                unique_count += 1

            # Create document entry
            doc_entry = {
                "rank": i + 1,
                "source": metadata.get('source', 'unknown'),
                "content_preview": content[:self.content_preview_length],
                "content_length": len(content),
                "score": scores[i] if scores and i < len(scores) else None,
                "is_duplicate": is_duplicate,
                "metadata": {k: str(v) for k, v in metadata.items()}  # Ensure serializable
            }
            self.current_entry.retrieved_documents.append(doc_entry)

        self.current_entry.unique_docs_retrieved = unique_count
        self.current_entry.duplicate_docs_count = len(documents) - unique_count

    def log_hyde(
        self,
        hypothetical_document: str,
        generation_time: Optional[float] = None
    ) -> None:
        """Log HyDE hypothetical document"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.hyde_enabled = True
        self.current_entry.hypothetical_document = hypothetical_document
        self.current_entry.hyde_generation_time = generation_time

    def log_self_rag_reflection(
        self,
        relevance_token: Optional[str] = None,
        relevance_score: Optional[float] = None,
        support_token: Optional[str] = None,
        support_score: Optional[float] = None,
        utility_token: Optional[str] = None,
        utility_score: Optional[float] = None,
        overall_score: Optional[float] = None,
        regeneration_count: int = 0,
        reflection_time: Optional[float] = None
    ) -> None:
        """Log Self-RAG reflection results"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.self_rag_enabled = True
        self.current_entry.relevance_token = relevance_token
        self.current_entry.relevance_score = relevance_score
        self.current_entry.support_token = support_token
        self.current_entry.support_score = support_score
        self.current_entry.utility_token = utility_token
        self.current_entry.utility_score = utility_score
        self.current_entry.overall_reflection_score = overall_score
        self.current_entry.regeneration_count = regeneration_count
        self.current_entry.reflection_time = reflection_time

    def log_graphrag(
        self,
        entities_used: List[str],
        relationships_used: List[str],
        num_hops: int = 0
    ) -> None:
        """Log GraphRAG details"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.graphrag_enabled = True
        self.current_entry.entities_used = entities_used
        self.current_entry.relationships_used = relationships_used
        self.current_entry.num_hops = num_hops

    def log_response(
        self,
        response: str,
        generation_time: Optional[float] = None,
        total_time: Optional[float] = None
    ) -> None:
        """Log LLM response"""
        if not self.enabled or not self.current_entry:
            return

        self.current_entry.llm_response = response
        self.current_entry.response_length = len(response)
        self.current_entry.generation_time = generation_time
        self.current_entry.total_time = total_time

    def log_error(self, error: str) -> None:
        """Log an error"""
        if not self.enabled or not self.current_entry:
            return
        self.current_entry.errors.append(error)

    def log_warning(self, warning: str) -> None:
        """Log a warning"""
        if not self.enabled or not self.current_entry:
            return
        self.current_entry.warnings.append(warning)

    def save_entry(self) -> Optional[str]:
        """Save the current entry to log files"""
        if not self.enabled or not self.current_entry:
            return None

        entry_dict = asdict(self.current_entry)

        # Save to JSON
        if self.save_format in ["json", "both"]:
            self._append_json_entry(entry_dict)

        # Save to TXT
        if self.save_format in ["txt", "both"]:
            self._append_txt_entry(entry_dict)

        # Clear current entry
        saved_entry_id = self.current_entry.entry_id
        self.current_entry = None

        return f"Entry {saved_entry_id} saved"

    def _append_json_entry(self, entry_dict: Dict) -> None:
        """Append entry to JSON log file"""
        # Read existing data
        with open(self.json_log_path, 'r') as f:
            data = json.load(f)

        # Append new entry
        data["entries"].append(entry_dict)

        # Write back
        with open(self.json_log_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _append_txt_entry(self, entry_dict: Dict) -> None:
        """Append entry to TXT log file in human-readable format"""
        with open(self.txt_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ENTRY #{entry_dict['entry_id']} - {entry_dict['timestamp']}\n")
            f.write("=" * 80 + "\n\n")

            # User Query
            f.write("USER QUERY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{entry_dict['user_query']}\n\n")

            # Query Analysis
            if entry_dict.get('selected_strategy'):
                f.write("QUERY ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Complexity Score: {entry_dict.get('complexity_score', 'N/A')}/10\n")
                f.write(f"  Complexity Level: {entry_dict.get('complexity_level', 'N/A')}\n")
                f.write(f"  Selected Strategy: {entry_dict.get('selected_strategy', 'N/A')}\n")
                if entry_dict.get('routing_reasoning'):
                    f.write(f"  Reasoning: {entry_dict['routing_reasoning']}\n")
                if entry_dict.get('query_analysis_time'):
                    f.write(f"  Analysis Time: {entry_dict['query_analysis_time']:.2f}s\n")
                f.write("\n")

            # Retrieval
            if entry_dict.get('total_docs_retrieved', 0) > 0:
                f.write("RETRIEVAL:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Total Retrieved: {entry_dict['total_docs_retrieved']}\n")
                f.write(f"  Unique Documents: {entry_dict['unique_docs_retrieved']}\n")
                f.write(f"  Duplicates: {entry_dict['duplicate_docs_count']}\n")
                if entry_dict.get('retrieval_time'):
                    f.write(f"  Retrieval Time: {entry_dict['retrieval_time']:.2f}s\n")
                f.write("\n")

                f.write("  Retrieved Documents:\n")
                for doc in entry_dict.get('retrieved_documents', [])[:5]:  # Show first 5
                    dup_marker = " [DUPLICATE]" if doc.get('is_duplicate') else ""
                    f.write(f"    #{doc['rank']}: {doc['source']}{dup_marker}\n")
                    f.write(f"        Length: {doc['content_length']} chars\n")
                    preview = doc['content_preview'][:200].replace('\n', ' ')
                    f.write(f"        Preview: {preview}...\n\n")

            # HyDE
            if entry_dict.get('hyde_enabled'):
                f.write("HYDE (Hypothetical Document Embeddings):\n")
                f.write("-" * 40 + "\n")
                hypo_doc = entry_dict.get('hypothetical_document', '')[:500]
                f.write(f"  Hypothetical Document:\n    {hypo_doc}...\n")
                if entry_dict.get('hyde_generation_time'):
                    f.write(f"  Generation Time: {entry_dict['hyde_generation_time']:.2f}s\n")
                f.write("\n")

            # Self-RAG Reflection
            if entry_dict.get('self_rag_enabled'):
                f.write("SELF-RAG REFLECTION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Relevance: {entry_dict.get('relevance_token', 'N/A')} (score: {entry_dict.get('relevance_score', 'N/A')})\n")
                f.write(f"  Support: {entry_dict.get('support_token', 'N/A')} (score: {entry_dict.get('support_score', 'N/A')})\n")
                f.write(f"  Utility: {entry_dict.get('utility_token', 'N/A')} (score: {entry_dict.get('utility_score', 'N/A')})\n")
                f.write(f"  Overall Score: {entry_dict.get('overall_reflection_score', 'N/A')}\n")
                f.write(f"  Regenerations: {entry_dict.get('regeneration_count', 0)}\n")
                if entry_dict.get('reflection_time'):
                    f.write(f"  Reflection Time: {entry_dict['reflection_time']:.2f}s\n")
                f.write("\n")

            # GraphRAG
            if entry_dict.get('graphrag_enabled'):
                f.write("GRAPHRAG:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Entities Used: {', '.join(entry_dict.get('entities_used', [])[:5])}\n")
                f.write(f"  Relationships: {len(entry_dict.get('relationships_used', []))}\n")
                f.write(f"  Hops: {entry_dict.get('num_hops', 0)}\n")
                f.write("\n")

            # LLM Response
            f.write("LLM RESPONSE:\n")
            f.write("-" * 40 + "\n")
            response = entry_dict.get('llm_response', 'No response')
            f.write(f"{response}\n\n")

            # Timing Summary
            f.write("TIMING:\n")
            f.write("-" * 40 + "\n")
            if entry_dict.get('generation_time'):
                f.write(f"  Generation Time: {entry_dict['generation_time']:.2f}s\n")
            if entry_dict.get('total_time'):
                f.write(f"  Total Time: {entry_dict['total_time']:.2f}s\n")
            f.write(f"  Response Length: {entry_dict.get('response_length', 0)} chars\n")

            # Errors/Warnings
            if entry_dict.get('errors'):
                f.write("\nERRORS:\n")
                for err in entry_dict['errors']:
                    f.write(f"  - {err}\n")

            if entry_dict.get('warnings'):
                f.write("\nWARNINGS:\n")
                for warn in entry_dict['warnings']:
                    f.write(f"  - {warn}\n")

            f.write("\n")

    def get_log_paths(self) -> Dict[str, str]:
        """Get paths to log files"""
        return {
            "json": str(self.json_log_path) if self.save_format in ["json", "both"] else None,
            "txt": str(self.txt_log_path) if self.save_format in ["txt", "both"] else None,
        }


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get or create the global debug logger"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger


def init_debug_logger(
    output_dir: str = "./debug_logs",
    session_id: Optional[str] = None,
    enabled: bool = True,
    save_format: str = "both"
) -> DebugLogger:
    """Initialize the global debug logger with custom settings"""
    global _debug_logger
    _debug_logger = DebugLogger(
        output_dir=output_dir,
        session_id=session_id,
        enabled=enabled,
        save_format=save_format
    )
    return _debug_logger


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("DEBUG LOGGER TEST")
    print("=" * 70)

    # Initialize logger
    logger = DebugLogger(output_dir="./debug_logs", session_id="test_session")

    # Simulate a query
    logger.start_entry("Can you give me a summary of the BERT paper?")

    logger.log_query_analysis(
        complexity_score=4,
        complexity_level="medium",
        selected_strategy="hyde",
        analysis_time=0.5
    )

    # Simulate retrieved documents
    class MockDoc:
        def __init__(self, content, source):
            self.page_content = content
            self.metadata = {"source": source}

    docs = [
        MockDoc("BERT uses masked language modeling...", "bert_paper.pdf"),
        MockDoc("The transformer architecture was introduced...", "attention.pdf"),
        MockDoc("BERT uses masked language modeling...", "bert_paper.pdf"),  # Duplicate
    ]

    logger.log_retrieval(docs, retrieval_time=1.2)

    logger.log_hyde(
        hypothetical_document="BERT is a bidirectional transformer model that...",
        generation_time=2.0
    )

    logger.log_response(
        response="BERT (Bidirectional Encoder Representations from Transformers) introduces...",
        generation_time=3.5,
        total_time=7.2
    )

    logger.save_entry()

    print(f"\nLog files created:")
    for fmt, path in logger.get_log_paths().items():
        if path:
            print(f"  {fmt}: {path}")

    print("\n" + "=" * 70)
    print("TEST PASSED!")
    print("=" * 70)
