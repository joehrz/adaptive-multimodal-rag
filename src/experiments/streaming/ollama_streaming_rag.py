"""
Streaming RAG Implementation with Ollama
Provides real-time token-by-token generation for better UX
"""

import requests
import json
import time
from typing import List, Iterator, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from langchain.schema import Document


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
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        ollama_url: str = "http://localhost:11434",
        verbose: bool = True
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.verbose = verbose

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
                timeout=300
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
            if self.verbose:
                print(f"Error: {error_msg}")

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
        on_token: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[StreamingProgress], None]] = None
    ) -> Iterator[StreamChunk]:
        """
        Stream a multi-stage RAG process with progress tracking

        Args:
            query: User query
            documents: Retrieved documents
            use_hyde: Whether to use HyDE for hypothetical generation
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

            # Stream HyDE generation with document context
            hyde_tokens = []
            for chunk in self.stream_hyde_generation(query, documents=documents, on_token=on_token):
                hyde_tokens.append(chunk.content)
                yield chunk

            yield StreamChunk(
                stage=StreamingStage.GENERATING_HYPOTHETICALS,
                content="\n[Hypothetical document complete]\n\n",
                metadata={"hyde_length": len("".join(hyde_tokens))},
                timestamp=time.time()
            )

        # Stage 3: Retrieval
        if on_progress:
            on_progress(StreamingProgress(
                stage=StreamingStage.RETRIEVING_DOCS,
                progress=0.5,
                message=f"Retrieved {len(documents)} documents",
                timestamp=time.time()
            ))

        yield StreamChunk(
            stage=StreamingStage.RETRIEVING_DOCS,
            content=f"\n[Retrieved {len(documents)} relevant documents]\n\n",
            metadata={"doc_count": len(documents)},
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
        final_documents = documents

        # If HyDE was used, add hypothetical document as additional context
        if use_hyde and hyde_tokens:
            hyde_content = "".join(hyde_tokens)
            if hyde_content.strip():
                # Create document from HyDE output
                from langchain.schema import Document
                hyde_doc = Document(
                    page_content=hyde_content,
                    metadata={"source": "HyDE_Analysis", "type": "hypothetical"}
                )
                # Add HyDE document to context
                final_documents = [hyde_doc] + documents

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
