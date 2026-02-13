"""
Streaming RAG Implementation with Ollama
Provides real-time token-by-token generation for better UX
"""

import requests
import json
import time
import logging
from typing import List, Iterator, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from langchain.schema import Document

# Import config system
try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import HyDE for real embedding-based retrieval
try:
    from src.experiments.hyde.ollama_hyde import OllamaHyDE
    HYDE_AVAILABLE = True
except ImportError:
    HYDE_AVAILABLE = False

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)


class StreamingStage(Enum):
    """Stages in the streaming RAG pipeline"""
    ANALYZING_QUERY = "analyzing_query"
    RETRIEVING_DOCS = "retrieving_docs"
    GENERATING_HYPOTHETICALS = "generating_hypotheticals"
    RUNNING_REFLECTION = "running_reflection"
    GENERATING_ANSWER = "generating_answer"
    COMPLETE = "complete"


@dataclass
class StreamChunk:
    """A chunk of streaming data"""
    stage: StreamingStage
    content: str
    metadata: dict
    timestamp: float


@dataclass
class StreamingProgress:
    """Progress update during streaming"""
    stage: StreamingStage
    progress: float # 0.0 to 1.0
    message: str
    timestamp: float


class OllamaStreamingRAG:
    """
    Streaming RAG implementation using Ollama's streaming API

    Supports:
    - Real-time token generation
    - Progress updates for multi-step processes
    - Stage-based streaming (query analysis, retrieval, generation)
    - Real HyDE embedding-based retrieval when use_hyde=True
    """

    def __init__(
        self,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        verbose: Optional[bool] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
    ):
        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.ollama_url = ollama_url if ollama_url is not None else config.ollama.url
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.timeout = timeout if timeout is not None else config.ollama.timeout
        else:
            self.model = model or "qwen2.5:14b"
            self.ollama_url = ollama_url or "http://localhost:11434"
            self.verbose = verbose if verbose is not None else True
            self.timeout = timeout or 300

        # Initialize HyDE engine for real embedding-based retrieval
        self._hyde_engine: Optional[OllamaHyDE] = None

    def _get_hyde_engine(self) -> Optional['OllamaHyDE']:
        """
        Lazily initialize and return HyDE engine for embedding-based retrieval.

        Returns:
            OllamaHyDE instance if available, None otherwise
        """
        if not HYDE_AVAILABLE:
            logger.warning("HyDE not available - falling back to text-based retrieval")
            return None

        if self._hyde_engine is None:
            try:
                self._hyde_engine = OllamaHyDE(
                    model=self.model,
                    verbose=self.verbose
                )
                logger.info("HyDE engine initialized for streaming RAG")
            except Exception as e:
                logger.error(f"Failed to initialize HyDE engine: {e}")
                return None

        return self._hyde_engine

    def stream_generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[StreamingProgress], None]] = None
    ) -> Iterator[StreamChunk]:
        """
        Stream generation from Ollama

        Args:
            prompt: Input prompt
            on_token: Callback for each token (optional)
            on_progress: Callback for progress updates (optional)

        Yields:
            StreamChunk objects with generated tokens
        """
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.GENERATING_ANSWER,
                progress=0.0,
                message="Starting generation...",
                timestamp=time.time()
            ))

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            full_response = ""
            token_count = 0

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)

                        if "response" in data:
                            token = data["response"]
                            full_response += token
                            token_count += 1

                            # Call token callback
                            if on_token:
                                on_token(token)

                            # Yield chunk
                            chunk = StreamChunk(
                                stage=StreamingStage.GENERATING_ANSWER,
                                content=token,
                                metadata={
                                    "token_count": token_count,
                                    "done": data.get("done", False)
                                },
                                timestamp=time.time()
                            )
                            yield chunk

                            # Progress update every 10 tokens
                            if on_progress and token_count % 10 == 0:
                                on_progress(StreamingProgress(
                                    stage=StreamingStage.GENERATING_ANSWER,
                                    progress=0.5, # Indeterminate
                                    message=f"Generated {token_count} tokens...",
                                    timestamp=time.time()
                                ))

                        if data.get("done", False):
                            if on_progress:
                                on_progress(StreamingProgress(
                                    stage=StreamingStage.COMPLETE,
                                    progress=1.0,
                                    message=f"Complete! Generated {token_count} tokens",
                                    timestamp=time.time()
                                ))
                            break

                    except json.JSONDecodeError:
                        continue

        except requests.RequestException as e:
            error_msg = f"Streaming error: {str(e)}"
            logger.error(error_msg)

            yield StreamChunk(
                stage=StreamingStage.COMPLETE,
                content="",
                metadata={"error": error_msg},
                timestamp=time.time()
            )

    def stream_rag_query(
        self,
        query: str,
        documents: List[Document],
        on_token: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[StreamingProgress], None]] = None
    ) -> Iterator[StreamChunk]:
        """
        Stream a complete RAG query with progress updates

        Args:
            query: User query
            documents: Retrieved documents for context
            on_token: Callback for each generated token
            on_progress: Callback for progress updates

        Yields:
            StreamChunk objects for the complete RAG pipeline
        """
        # Stage 1: Preparing context
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.RETRIEVING_DOCS,
                progress=0.2,
                message=f"Using {len(documents)} retrieved documents",
                timestamp=time.time()
            ))

        # Build context from documents
        context = "\n\n".join([
            f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:2000]}"
            for i, doc in enumerate(documents[:5]) # Use up to 5 documents
        ])

        # Stage 2: Generate answer with streaming
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.GENERATING_ANSWER,
                progress=0.4,
                message="Generating answer...",
                timestamp=time.time()
            ))

        prompt = f"""Based on the following context documents, provide a comprehensive answer to the question. If the first document contains a hypothetical analysis, use it to inform your response while also incorporating information from the other retrieved documents.

Context:
{context}

Question: {query}

Provide a detailed answer that synthesizes information from all the context documents:"""

        # Stream the generation
        yield from self.stream_generate(
            prompt=prompt,
            on_token=on_token,
            on_progress=on_progress
        )

    def stream_hyde_generation(
        self,
        query: str,
        documents: List[Document] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[StreamingProgress], None]] = None
    ) -> Iterator[StreamChunk]:
        """
        Stream HyDE hypothetical document generation

        Args:
            query: User query
            documents: Available documents to inform hypothetical generation
            on_token: Callback for each token
            on_progress: Callback for progress updates

        Yields:
            StreamChunk objects with hypothetical document tokens
        """
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.GENERATING_HYPOTHETICALS,
                progress=0.0,
                message="Generating hypothetical document...",
                timestamp=time.time()
            ))

        # Create context-aware prompt if documents available
        if documents:
            # Get document sources and preview content for better context
            doc_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in documents[:5]]))

            # Get a sample of actual content for context
            content_previews = []
            for doc in documents[:3]:
                preview = doc.page_content[:200].replace('\n', ' ').strip()
                if preview:
                    content_previews.append(f"- {doc.metadata.get('source', 'Unknown')}: {preview}...")

            doc_context = f"""Based on the specific uploaded documents: {', '.join(doc_sources)}

Key content includes:
{chr(10).join(content_previews)}

"""
        else:
            doc_context = ""

        prompt = f"""You are analyzing specific research papers. Generate a detailed document that directly answers the question about the relationships between these specific papers.

{doc_context}
Focus on the actual papers mentioned above, their specific contributions, methodologies, and how they relate to each other.

Question: {query}

Analysis Document:"""

        # Update stage in chunks
        for chunk in self.stream_generate(prompt=prompt, on_token=on_token):
            chunk.stage = StreamingStage.GENERATING_HYPOTHETICALS
            yield chunk

        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.GENERATING_HYPOTHETICALS,
                progress=1.0,
                message="Hypothetical document complete",
                timestamp=time.time()
            ))

    def stream_multi_stage_rag(
        self,
        query: str,
        documents: List[Document],
        use_hyde: bool = False,
        vector_store: Optional[any] = None,
        top_k: int = 5,
        on_token: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[StreamingProgress], None]] = None
    ) -> Iterator[StreamChunk]:
        """
        Stream a multi-stage RAG process with progress tracking

        Args:
            query: User query
            documents: Retrieved documents (used if vector_store not provided)
            use_hyde: Whether to use HyDE for hypothetical generation/retrieval
            vector_store: Optional vector store for real HyDE embedding retrieval
            top_k: Number of documents to retrieve with HyDE (default 5)
            on_token: Token callback
            on_progress: Progress callback

        Yields:
            StreamChunk objects for all stages
        """
        # Stage 1: Query Analysis
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.ANALYZING_QUERY,
                progress=0.1,
                message="Analyzing query complexity...",
                timestamp=time.time()
            ))

        yield StreamChunk(
            stage=StreamingStage.ANALYZING_QUERY,
            content=f"[Analyzing query: {query[:50]}...]\n",
            metadata={"query_length": len(query)},
            timestamp=time.time()
        )

        # Stage 2: HyDE (if enabled)
        hyde_tokens = []
        hyde_retrieved_docs = None

        if use_hyde:
            if on_progress:
                on_progress(StreamingProgress(
                    stage=StreamingStage.GENERATING_HYPOTHETICALS,
                    progress=0.3,
                    message="Generating hypothetical document...",
                    timestamp=time.time()
                ))

            yield StreamChunk(
                stage=StreamingStage.GENERATING_HYPOTHETICALS,
                content="\n[Generating hypothetical document...]\n",
                metadata={},
                timestamp=time.time()
            )

            # Check if we can use real HyDE embedding-based retrieval
            hyde_engine = self._get_hyde_engine()

            if hyde_engine is not None and vector_store is not None:
                # Use real HyDE embedding-based retrieval
                try:
                    if on_progress:
                        on_progress(StreamingProgress(
                            stage=StreamingStage.GENERATING_HYPOTHETICALS,
                            progress=0.35,
                            message="Using HyDE embedding-based retrieval...",
                            timestamp=time.time()
                        ))

                    # Generate hypothetical document and retrieve with embeddings
                    hyde_result = hyde_engine.retrieve_with_hyde(
                        query=query,
                        vector_store=vector_store,
                        top_k=top_k
                    )

                    # Store the HyDE-retrieved documents
                    hyde_retrieved_docs = hyde_result.documents

                    # Stream the hypothetical document content for UX
                    hypothetical_content = hyde_result.hypothetical_document
                    for i in range(0, len(hypothetical_content), 10):
                        token = hypothetical_content[i:i+10]
                        hyde_tokens.append(token)
                        if on_token:
                            on_token(token)
                        yield StreamChunk(
                            stage=StreamingStage.GENERATING_HYPOTHETICALS,
                            content=token,
                            metadata={"hyde_mode": "embedding_retrieval"},
                            timestamp=time.time()
                        )

                    logger.info(f"HyDE embedding retrieval returned {len(hyde_retrieved_docs)} documents")

                except Exception as e:
                    logger.warning(f"HyDE embedding retrieval failed, falling back to streaming: {e}")
                    hyde_engine = None  # Fall back to streaming generation

            if hyde_engine is None or vector_store is None:
                # Fall back to streaming HyDE generation (text-based, no embedding retrieval)
                for chunk in self.stream_hyde_generation(query, documents=documents, on_token=on_token):
                    hyde_tokens.append(chunk.content)
                    yield chunk

            yield StreamChunk(
                stage=StreamingStage.GENERATING_HYPOTHETICALS,
                content="\n[Hypothetical document complete]\n\n",
                metadata={"hyde_length": len("".join(hyde_tokens)), "used_embedding_retrieval": hyde_retrieved_docs is not None},
                timestamp=time.time()
            )

        # Stage 3: Retrieval
        retrieval_count = len(hyde_retrieved_docs) if hyde_retrieved_docs is not None else len(documents)
        retrieval_method = "HyDE embedding retrieval" if hyde_retrieved_docs is not None else "standard retrieval"

        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.RETRIEVING_DOCS,
                progress=0.5,
                message=f"Retrieved {retrieval_count} documents via {retrieval_method}",
                timestamp=time.time()
            ))

        yield StreamChunk(
            stage=StreamingStage.RETRIEVING_DOCS,
            content=f"\n[Retrieved {retrieval_count} relevant documents via {retrieval_method}]\n\n",
            metadata={"doc_count": retrieval_count, "retrieval_method": retrieval_method},
            timestamp=time.time()
        )

        # Stage 4: Final Generation
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.GENERATING_ANSWER,
                progress=0.7,
                message="Generating final answer...",
                timestamp=time.time()
            ))

        yield StreamChunk(
            stage=StreamingStage.GENERATING_ANSWER,
            content="[Generating answer...]\n\n",
            metadata={},
            timestamp=time.time()
        )

        # Prepare final context
        # Use HyDE-retrieved documents if available, otherwise use provided documents
        if hyde_retrieved_docs is not None:
            # Use documents retrieved via HyDE embedding similarity
            final_documents = hyde_retrieved_docs
            logger.info(f"Using {len(final_documents)} HyDE-retrieved documents for context")
        else:
            final_documents = documents

        # If HyDE was used (either mode), add hypothetical document as additional context
        if use_hyde and hyde_tokens:
            hyde_content = "".join(hyde_tokens)
            if hyde_content.strip():
                # Create document from HyDE output
                hyde_doc = Document(
                    page_content=hyde_content,
                    metadata={"source": "HyDE_Analysis", "type": "hypothetical"}
                )
                # Add HyDE document to context (prepend for priority)
                final_documents = [hyde_doc] + final_documents

        # Stream final answer with enhanced context
        yield from self.stream_rag_query(
            query=query,
            documents=final_documents,
            on_token=on_token,
            on_progress=on_progress
        )


class StreamingConsoleDisplay:
    """Helper class for displaying streaming output in console"""

    def __init__(self, show_progress: bool = True, show_tokens: bool = True):
        self.show_progress = show_progress
        self.show_tokens = show_tokens
        self.current_stage = None

    def on_token(self, token: str):
        """Display token in real-time"""
        if self.show_tokens:
            print(token, end="", flush=True)

    def on_progress(self, progress: StreamingProgress):
        """Display progress updates"""
        if self.show_progress and progress.stage != self.current_stage:
            self.current_stage = progress.stage
            progress_bar = "=" * int(progress.progress * 30)
            remaining = " " * (30 - int(progress.progress * 30))
            print(f"\n[{progress_bar}{remaining}] {progress.message}", flush=True)


if __name__ == "__main__":
    print("=" * 80)
    print("STREAMING RAG TEST")
    print("=" * 80)

    # Test basic streaming
    streaming_rag = OllamaStreamingRAG(model="qwen2.5:14b", verbose=True)
    display = StreamingConsoleDisplay(show_progress=True, show_tokens=True)

    test_query = "What is machine learning?"

    print(f"\nQuery: {test_query}")
    print("\nStreaming response:\n")

    # Simple streaming test
    for chunk in streaming_rag.stream_generate(
        prompt=test_query,
        on_token=display.on_token,
        on_progress=display.on_progress
    ):
        pass # Callbacks handle display

    print("\n\n" + "=" * 80)
    print("Streaming test complete!")
