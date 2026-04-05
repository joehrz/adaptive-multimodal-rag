"""
Ollama-powered RAG implementation
Uses open-source LLMs (Llama, Mistral, etc.) running locally via Ollama
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from src.core.caching_system import RAGCacheManager
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

from src.core.retrieval import (
    deduplicate_chunks,
    deduplicate_documents,
    detect_page_query,
    detect_metadata_query,
    detect_summarization_query,
    keyword_search,
    retrieve_first_pages,
    rerank,
)
from src.core.generation import (
    build_prompt,
    generate_response,
    format_conversation_history,
)

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)


class OllamaRAG:
    """
    RAG implementation using Ollama for local open-source LLM generation
    Supports Llama 2, Mistral, CodeLlama, and other models
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        k_retrieval: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        persist_directory: Optional[str] = None,
        verbose: Optional[bool] = None,
        enable_caching: Optional[bool] = None,
        cache_manager: Optional['RAGCacheManager'] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        enable_reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
    ):
        """
        Initialize Ollama RAG.

        Args:
            model: Ollama model name (default from config or "qwen2.5:14b")
            embedding_model: Embedding model name
            temperature: Generation temperature
            max_tokens: Max tokens per response
            k_retrieval: Number of documents to retrieve
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            persist_directory: Vector store persist path
            verbose: Enable verbose logging
            enable_caching: Enable query response caching
            cache_manager: Optional RAGCacheManager for caching
            config: Optional Config object
            timeout: Timeout for LLM calls in seconds
        """

        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.temperature = temperature if temperature is not None else config.llm.temperature
            self.max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
            self.k_retrieval = k_retrieval if k_retrieval is not None else config.documents.k_retrieval
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.timeout = timeout if timeout is not None else config.llm.timeout
            self.persist_directory = persist_directory if persist_directory is not None else config.vector_db.persist_directory
            self.dedup_min_chars = config.documents.dedup_min_chars
            _embedding_model = embedding_model if embedding_model is not None else config.embeddings.model
            _embedding_device = config.embeddings.device
            _chunk_size = chunk_size if chunk_size is not None else config.documents.chunk_size
            _chunk_overlap = chunk_overlap if chunk_overlap is not None else config.documents.chunk_overlap
            _enable_caching = enable_caching if enable_caching is not None else config.cache.enabled
            _enable_reranker = enable_reranker if enable_reranker is not None else config.reranker.enabled
            _reranker_model = reranker_model if reranker_model is not None else config.reranker.model
            _reranker_device = config.reranker.device
            self.reranker_top_k = config.reranker.top_k
            self.reranker_candidates = config.reranker.candidates
            self.reranker_min_score = config.reranker.min_score
        else:
            self.model = model or "qwen2.5:14b"
            self.temperature = temperature if temperature is not None else 0.3
            self.max_tokens = max_tokens or 1000
            self.k_retrieval = k_retrieval or 10
            self.verbose = verbose if verbose is not None else True
            self.timeout = timeout or 120
            self.persist_directory = persist_directory or "./data/chroma_db_ollama"
            self.dedup_min_chars = 500
            _embedding_model = embedding_model or "all-MiniLM-L6-v2"
            _embedding_device = "cpu"
            _chunk_size = chunk_size or 1000
            _chunk_overlap = chunk_overlap or 200
            _enable_caching = enable_caching if enable_caching is not None else True
            _enable_reranker = enable_reranker if enable_reranker is not None else True
            _reranker_model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            _reranker_device = "cpu"
            self.reranker_top_k = 10
            self.reranker_candidates = 30
            self.reranker_min_score = 5.0

        # Create Ollama client with timeout
        self._ollama_client = ollama.Client(timeout=self.timeout)

        # Test Ollama connection
        try:
            available_models = self._ollama_client.list()
            model_names = [m.model for m in available_models.models]

            if self.model not in model_names:
                if self.verbose:
                    logger.warning(f"Model {self.model} not found locally")
                    logger.warning(f"Available models: {model_names}")
                    logger.warning(f"Download with: ollama pull {self.model}")
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")

            if self.verbose:
                logger.info(f"Ollama connected with model: {self.model}")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

        # Initialize embeddings
        from src.core.embeddings import get_embeddings
        self.embeddings = get_embeddings(
            config=config.embeddings if config else None,
            model=_embedding_model,
            device=_embedding_device,
        )

        # Initialize cross-encoder reranker
        self.reranker = None
        if _enable_reranker and RERANKER_AVAILABLE:
            try:
                self.reranker = CrossEncoder(
                    _reranker_model,
                    device=_reranker_device
                )
                if self.verbose:
                    logger.info(f"Reranker: {_reranker_model} on {_reranker_device}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                self.reranker = None
        elif _enable_reranker and not RERANKER_AVAILABLE:
            if self.verbose:
                logger.warning("Reranker requested but sentence-transformers not installed")

        # Initialize text splitter
        _chunking_strategy = config.documents.chunking_strategy if config else "recursive"
        if _chunking_strategy == "semantic":
            from src.core.chunking import SemanticChunker
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                min_chunk_size=200,
                max_chunk_size=_chunk_size * 2,
            )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap
            )

        self.vector_store = None
        self.documents = []

        # Initialize caching with semantic similarity matching
        self.enable_caching = _enable_caching and CACHING_AVAILABLE
        if self.enable_caching:
            if cache_manager:
                self.cache_manager = cache_manager
            else:
                def _embed_query(text: str) -> List[float]:
                    return self.embeddings.embed_query(text)

                self.cache_manager = RAGCacheManager(
                    enable_auto_cleanup=True,
                    embed_fn=_embed_query,
                )
            if self.verbose:
                logger.info("Caching: Enabled (semantic similarity)")
        else:
            self.cache_manager = None
            if self.verbose:
                logger.info("Caching: Disabled")

        if self.verbose:
            logger.info(f"Ollama RAG initialized: model={self.model}, embedding={_embedding_model}, chunk_size={_chunk_size}")

    # ── Document Management ──────────────────────────────────────────────

    def add_documents(self, documents: List[Document], deduplicate: bool = True):
        """Add documents to the vector store with optional deduplication."""
        if not documents:
            if self.verbose:
                logger.warning("No documents provided to add_documents()")
            return

        self.documents.extend(documents)

        if self.verbose:
            logger.info(f"Processing {len(documents)} documents...")

        chunks = self.text_splitter.split_documents(documents)
        original_chunk_count = len(chunks)

        if deduplicate:
            chunks = deduplicate_chunks(chunks)
            if self.verbose:
                removed = original_chunk_count - len(chunks)
                if removed > 0:
                    logger.info(f"[DEDUP] Removed {removed} duplicate chunks during ingestion")

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(chunks)

        if self.verbose:
            logger.info(f"Added {len(chunks)} unique chunks to vector store")

    def clear_vector_store(self) -> bool:
        """Clear the vector store completely, removing all persisted data."""
        import shutil

        try:
            if self.vector_store is not None:
                try:
                    self.vector_store.delete_collection()
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not delete collection: {e}")
                self.vector_store = None

            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    logger.info(f"Deleted persist directory: {self.persist_directory}")

            self.documents = []
            self.clear_cache()

            if self.verbose:
                logger.info("Vector store cleared successfully")

            return True

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    # ── Retrieval ────────────────────────────────────────────────────────

    def _retrieve_documents(self, query: str, k: int = None, bypass_cache: bool = False) -> List[Document]:
        """Retrieve relevant documents with deduplication and smart query detection."""
        k = k if k is not None else self.k_retrieval

        if not self.vector_store:
            if self.verbose:
                logger.warning("No documents in vector store - retrieval will return empty results")
            return []

        # Check for metadata queries (title, authors, etc.)
        if detect_metadata_query(query):
            if self.verbose:
                logger.info("[METADATA QUERY] Retrieving first pages for metadata question")
            first_pages = retrieve_first_pages(self.documents, self.vector_store, verbose=self.verbose)
            if first_pages:
                semantic_docs = self.vector_store.similarity_search(query, k=k)
                all_docs = first_pages + semantic_docs
                return deduplicate_documents(all_docs, self.dedup_min_chars)[:k]

        # Check if this is a page-specific query
        page_num = detect_page_query(query)

        if page_num is not None:
            if self.verbose:
                logger.info(f"[PAGE QUERY] Detected request for page {page_num}")
            try:
                docs = self.vector_store.similarity_search(
                    query, k=k * 3, filter={"page": page_num}
                )
                if docs:
                    docs = deduplicate_documents(docs, self.dedup_min_chars)[:k]
                    if self.verbose:
                        logger.info(f"[PAGE FILTER] Found {len(docs)} unique documents for page {page_num}")
                    return docs
                else:
                    if self.verbose:
                        logger.info(f"[PAGE FILTER] No documents found for page {page_num}, falling back to semantic search")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"[PAGE FILTER] Metadata filter failed: {e}, falling back to semantic search")

        # Perform search
        if self.reranker:
            raw_k = max(self.reranker_candidates, k * 3)
        else:
            raw_k = k * 3
        raw_docs = self.vector_store.similarity_search(query, k=raw_k)

        if self.verbose:
            logger.info(f"[RETRIEVAL] Semantic search returned {len(raw_docs)} documents")

        # Supplement with keyword-based retrieval
        kw_docs = keyword_search(self.documents, query, k=k)
        if kw_docs:
            raw_docs = raw_docs + kw_docs
            if self.verbose:
                logger.info(f"[RETRIEVAL] Keyword search added {len(kw_docs)} supplemental documents")

        docs = deduplicate_documents(raw_docs, self.dedup_min_chars)

        if self.reranker and len(docs) > 1:
            docs = rerank(query, docs, self.reranker, top_k=self.reranker_top_k,
                         min_score_gap=self.reranker_min_score, verbose=self.verbose)
        else:
            docs = docs[:k]

        if self.verbose:
            logger.info(f"[RETRIEVAL] Returning {len(docs)} documents" +
                        (" (reranked)" if self.reranker else " (after deduplication)"))

        return docs

    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Public method to retrieve deduplicated documents."""
        return self._retrieve_documents(query, k=k)

    # ── Generation ───────────────────────────────────────────────────────

    def _generate_response(self, query: str, context: str = "", require_citations: bool = True, conversation_history: list = None) -> str:
        """Generate response using Ollama with query-type-aware prompts."""
        is_summarization = detect_summarization_query(query)
        prompt = build_prompt(
            query, context, is_summarization=is_summarization,
            require_citations=require_citations,
            conversation_history=conversation_history,
        )
        return generate_response(
            self._ollama_client, self.model, prompt,
            temperature=self.temperature, max_tokens=self.max_tokens,
            verbose=self.verbose,
        )

    # ── Query ────────────────────────────────────────────────────────────

    def query(self, question: str, use_retrieval: bool = True, bypass_cache: bool = False, conversation_history: list = None) -> str:
        """Query the RAG system."""
        if self.verbose:
            logger.info(f"OLLAMA RAG QUERY: {question[:60]}... | Retrieval: {use_retrieval} | Caching: {self.cache_manager is not None}")

        # Check cache
        if self.cache_manager and not bypass_cache:
            cached = self.cache_manager.get_query_response(question)
            if cached:
                if self.verbose:
                    logger.info(f"[CACHE HIT] Returning cached response from {cached.get('cached_at', 'unknown')}")
                return cached["response"]

        context = ""
        docs = []
        is_summarization = detect_summarization_query(question)

        if use_retrieval:
            if is_summarization:
                docs = self._retrieve_documents(question)
                intro_docs = self._retrieve_documents("introduction background motivation")
                conclusion_docs = self._retrieve_documents("conclusion results findings contributions")
                all_docs = docs + intro_docs + conclusion_docs
                docs = deduplicate_documents(all_docs, self.dedup_min_chars)[:self.k_retrieval * 2]
            else:
                docs = self._retrieve_documents(question)

            if docs:
                max_chars_per_doc = 2000 if is_summarization else 1000
                context = "\n\n".join([
                    f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', '?')}): {doc.page_content[:max_chars_per_doc]}"
                    for i, doc in enumerate(docs)
                ])

                sources = set(doc.metadata.get('source', '') for doc in docs)
                if len(sources) > 1:
                    source_list = ", ".join(sorted(sources))
                    context = f"Note: The following context contains excerpts from multiple documents: {source_list}. Only use information from the document(s) relevant to the question.\n\n{context}"

                if self.verbose:
                    logger.info(f"Using {len(docs)} documents as context ({len(context)} chars)")
            else:
                if self.verbose:
                    logger.warning("No relevant documents found, using direct generation")
                use_retrieval = False

        try:
            answer = self._generate_response(question, context, conversation_history=conversation_history)
        except RuntimeError as e:
            logger.error(f"Generation failed for query: {question[:60]}...: {e}")
            return f"I'm sorry, I was unable to generate a response. Please try again. (Error: {e})"

        # Cache the response
        if self.cache_manager:
            doc_contents = [doc.page_content[:200] for doc in docs] if docs else []
            self.cache_manager.cache_query_response(
                query=question,
                response=answer,
                documents=doc_contents,
                strategy="baseline",
                metadata={"use_retrieval": use_retrieval, "doc_count": len(docs)}
            )
            if self.verbose:
                logger.info("[CACHED] Response cached for future queries")

        return answer

    def query_with_verification(self, question: str) -> Dict[str, Any]:
        """Query with verification - compares retrieval vs no-retrieval answers."""
        if self.verbose:
            logger.info(f"RAG VERIFICATION MODE: {question[:60]}...")

        is_summarization = detect_summarization_query(question)
        docs = self._retrieve_documents(question, bypass_cache=True)

        if is_summarization:
            intro_docs = self._retrieve_documents("introduction background motivation", bypass_cache=True)
            conclusion_docs = self._retrieve_documents("conclusion results findings contributions", bypass_cache=True)
            all_docs = docs + intro_docs + conclusion_docs
            docs = deduplicate_documents(all_docs, self.dedup_min_chars)[:self.k_retrieval * 2]

        context = ""
        retrieved_docs = []

        if docs:
            max_chars = 2000 if is_summarization else 1000
            context = "\n\n".join([
                f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content[:max_chars]}"
                for i, doc in enumerate(docs)
            ])
            retrieved_docs = [
                {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', ''),
                    "content_preview": doc.page_content[:500],
                    "full_length": len(doc.page_content)
                }
                for doc in docs
            ]

        answer_with_retrieval = self._generate_response(question, context, require_citations=True)
        answer_without_retrieval = self._generate_response(question, context="", require_citations=False)

        verification_notes = []

        if "cannot find" in answer_with_retrieval.lower() or "not found in" in answer_with_retrieval.lower():
            verification_notes.append("RAG answer indicates information not found in documents - may need better retrieval")

        if "[Document" in answer_with_retrieval:
            verification_notes.append("RAG answer includes citations - good sign retrieval is being used")

        if len(answer_with_retrieval) > 0 and len(answer_without_retrieval) > 0:
            words_rag = set(answer_with_retrieval.lower().split())
            words_llm = set(answer_without_retrieval.lower().split())
            overlap = len(words_rag & words_llm) / max(len(words_rag | words_llm), 1)

            if overlap > 0.8:
                verification_notes.append(f"WARNING: Answers are very similar ({overlap:.0%} overlap) - LLM may be using training knowledge")
            else:
                verification_notes.append(f"Answers differ significantly ({overlap:.0%} overlap) - retrieval likely providing unique info")

        if self.verbose:
            logger.debug(f"Retrieved context length: {len(context)} chars")
            for note in verification_notes:
                logger.info(f"Verification: {note}")

        return {
            "answer_with_retrieval": answer_with_retrieval,
            "answer_without_retrieval": answer_without_retrieval,
            "retrieved_context": context,
            "retrieved_docs": retrieved_docs,
            "verification_notes": verification_notes,
            "context_length": len(context),
            "num_docs_retrieved": len(docs)
        }

    # ── Utilities ────────────────────────────────────────────────────────

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return None

    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all()
            if self.verbose:
                logger.info("Cache cleared")

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions."""
        results = []
        for i, question in enumerate(questions, 1):
            if self.verbose:
                logger.info(f"Batch query {i}/{len(questions)}: {question[:50]}...")

            start_time = time.time()
            answer = self.query(question)
            processing_time = time.time() - start_time

            results.append({
                'question': question,
                'answer': answer,
                'processing_time': processing_time
            })
        return results

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            models = ollama.list()
            return [m.model for m in models.models]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        available_models = self.get_available_models()

        if new_model not in available_models:
            logger.error(f"Model {new_model} not available. Available: {available_models}")
            return False

        self.model = new_model
        if self.verbose:
            logger.info(f"Switched to model: {new_model}")
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model."""
        try:
            info = ollama.show(self.model)
            return {
                'name': self.model,
                'size': info.get('size', 'Unknown'),
                'family': info.get('details', {}).get('family', 'Unknown'),
                'parameters': info.get('details', {}).get('parameter_size', 'Unknown'),
                'quantization': info.get('details', {}).get('quantization_level', 'Unknown')
            }
        except Exception as e:
            return {'name': self.model, 'error': str(e)}

    # Keep backwards compatibility — these are used by other modules and tests
    _deduplicate_documents = staticmethod(deduplicate_documents)
    _deduplicate_chunks = staticmethod(deduplicate_chunks)
    _detect_summarization_query = staticmethod(detect_summarization_query)
    _detect_page_query = staticmethod(detect_page_query)
    _detect_metadata_query = staticmethod(detect_metadata_query)
    _format_conversation_history = staticmethod(format_conversation_history)

    def _keyword_search(self, query: str, k: int = 5) -> List[Document]:
        return keyword_search(self.documents, query, k=k)

    def _rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        return rerank(query, documents, self.reranker, top_k=top_k,
                     min_score_gap=self.reranker_min_score, verbose=self.verbose)

    def _retrieve_first_pages(self, num_pages: int = 2) -> List[Document]:
        return retrieve_first_pages(self.documents, self.vector_store, num_pages=num_pages, verbose=self.verbose)
