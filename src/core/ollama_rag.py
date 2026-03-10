"""
Ollama-powered RAG implementation
Uses open-source LLMs (Llama, Mistral, etc.) running locally via Ollama
"""

import os
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Import caching system
try:
    from src.core.caching_system import RAGCacheManager
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Import config system
try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

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
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        k_retrieval: Optional[int] = None,
        verbose: Optional[bool] = None,
        cache_manager: Optional['RAGCacheManager'] = None,
        enable_caching: Optional[bool] = None,
        persist_directory: Optional[str] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        enable_reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
    ):
        """
        Initialize Ollama RAG system

        Args:
            model: Ollama model to use (qwen2.5:14b, qwen2.5:7b, llama3.1:8b, etc.)
            embedding_model: Embedding model for retrieval
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens for generation
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            k_retrieval: Number of documents to retrieve
            verbose: Enable verbose logging
            cache_manager: Optional RAGCacheManager for caching
            enable_caching: Enable caching (default True)
            persist_directory: Directory for vector store persistence
            config: Optional Config object (uses global config if not provided)
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
        else:
            # Fallback to hardcoded defaults if no config available
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

        # Test Ollama connection
        try:
            available_models = ollama.list()
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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=_embedding_model,
            model_kwargs={'device': _embedding_device}
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap
        )

        self.vector_store = None
        self.documents = []

        # Initialize caching
        self.enable_caching = _enable_caching and CACHING_AVAILABLE
        if self.enable_caching:
            if cache_manager:
                self.cache_manager = cache_manager
            else:
                self.cache_manager = RAGCacheManager(enable_auto_cleanup=True)
            if self.verbose:
                logger.info("Caching: Enabled")
        else:
            self.cache_manager = None
            if self.verbose:
                logger.info("Caching: Disabled")

        if self.verbose:
            logger.info(f"Ollama RAG initialized: model={self.model}, embedding={_embedding_model}, chunk_size={_chunk_size}")

    def add_documents(self, documents: List[Document], deduplicate: bool = True):
        """
        Add documents to the vector store with optional deduplication.

        Args:
            documents: List of documents to add
            deduplicate: Whether to deduplicate chunks before adding (default True)
        """

        if not documents:
            if self.verbose:
                logger.warning("No documents provided to add_documents()")
            return

        self.documents.extend(documents)

        if self.verbose:
            logger.info(f"Processing {len(documents)} documents...")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        original_chunk_count = len(chunks)

        # Deduplicate chunks before adding to vector store
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
            if self.verbose:
                removed = original_chunk_count - len(chunks)
                if removed > 0:
                    logger.info(f"[DEDUP] Removed {removed} duplicate chunks during ingestion")

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(chunks)

        if self.verbose:
            logger.info(f"Added {len(chunks)} unique chunks to vector store")

    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Deduplicate chunks before adding to vector store.
        This prevents the same content from being indexed multiple times.

        Args:
            chunks: List of document chunks

        Returns:
            List of unique chunks
        """
        if not chunks:
            return []

        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            content = chunk.page_content.strip()
            if not content:
                continue

            # Create hash of full content for ingestion dedup
            content_hash = hashlib.sha256(content.lower().encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def _detect_page_query(self, query: str) -> Optional[int]:
        """Detect if query is asking about a specific page number"""
        import re
        # Patterns like "page 3", "page 5", "on page 10", "summarize page 3"
        patterns = [
            r'page\s*(\d+)',
            r'p\.?\s*(\d+)',
            r'pg\.?\s*(\d+)',
        ]
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        return None

    def _deduplicate_documents(self, docs: List[Document], min_chars: int = None) -> List[Document]:
        """
        Deduplicate a list of documents based on content hash.

        Args:
            docs: List of documents to deduplicate
            min_chars: Minimum characters to use for hashing (uses self.dedup_min_chars if None)

        Returns:
            List of unique documents
        """
        if not docs:
            return []

        min_chars = min_chars or self.dedup_min_chars
        seen_content = set()
        unique_docs = []

        for doc in docs:
            # Normalize content for comparison (strip whitespace, lowercase for hash)
            content = doc.page_content.strip()
            if not content:
                continue

            # Use configurable minimum characters for robust deduplication
            dedup_chars = min(min_chars, len(content))
            # Hash normalized content
            content_for_hash = content[:dedup_chars].lower()
            content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def _detect_metadata_query(self, query: str) -> bool:
        """Detect if query is asking about document metadata (title, authors, date, etc.)"""
        metadata_keywords = [
            'title of', 'paper title', 'what is the title',
            'who wrote', 'who are the authors', 'authors of',
            'when was .* published', 'publication date', 'year of publication',
            'which journal', 'which conference', 'where was .* published',
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in metadata_keywords)

    def _retrieve_first_pages(self, num_pages: int = 2) -> List[Document]:
        """Retrieve chunks from the first pages of uploaded documents (for metadata queries)"""
        first_page_docs = []

        # Search stored documents directly for first-page content
        for doc in self.documents:
            page = doc.metadata.get('page')
            if page is not None and page <= num_pages:
                first_page_docs.append(doc)

        if first_page_docs:
            if self.verbose:
                logger.info(f"[METADATA] Found {len(first_page_docs)} chunks from first {num_pages} pages")
            return self._deduplicate_documents(first_page_docs)

        # Fallback: if no page metadata, search vector store for abstract/introduction
        if self.vector_store:
            try:
                first_page_docs = self.vector_store.similarity_search(
                    "abstract introduction title authors",
                    k=5
                )
            except Exception:
                pass

        return self._deduplicate_documents(first_page_docs)

    def _keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """Supplement semantic search with keyword-based retrieval from stored documents"""
        if not self.documents:
            return []

        query_terms = [w.lower() for w in query.split() if len(w) > 3]
        if not query_terms:
            return []

        scored_chunks = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # Count how many query terms appear in this document
            matches = sum(1 for term in query_terms if term in content_lower)
            if matches > 0:
                scored_chunks.append((matches, doc))

        # Sort by match count descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_chunks[:k]]

    def _rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Rerank documents using cross-encoder model for more accurate relevance scoring."""
        if not documents or not self.reranker:
            return documents[:top_k]

        # Build query-document pairs for the cross-encoder
        pairs = [(query, doc.page_content[:512]) for doc in documents]

        try:
            scores = self.reranker.predict(pairs)

            # Sort documents by cross-encoder score (descending)
            scored_docs = sorted(
                zip(scores, documents),
                key=lambda x: x[0],
                reverse=True
            )

            reranked = [doc for score, doc in scored_docs[:top_k]]

            if self.verbose:
                top_score = scored_docs[0][0] if scored_docs else 0
                bottom_score = scored_docs[min(top_k - 1, len(scored_docs) - 1)][0] if scored_docs else 0
                logger.info(f"[RERANK] Reranked {len(documents)} -> {len(reranked)} docs "
                           f"(scores: {top_score:.3f} to {bottom_score:.3f})")

            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Falling back to original order.")
            return documents[:top_k]

    def _retrieve_documents(self, query: str, bypass_cache: bool = False) -> List[Document]:
        """Retrieve relevant documents with caching, deduplication, and smart query detection"""

        if not self.vector_store:
            if self.verbose:
                logger.warning("No documents in vector store - retrieval will return empty results")
            return []

        # Check for metadata queries (title, authors, etc.) - retrieve first pages
        if self._detect_metadata_query(query):
            if self.verbose:
                logger.info(f"[METADATA QUERY] Retrieving first pages for metadata question")
            first_pages = self._retrieve_first_pages()
            if first_pages:
                # Also do semantic search and merge
                semantic_docs = self.vector_store.similarity_search(query, k=self.k_retrieval)
                all_docs = first_pages + semantic_docs
                return self._deduplicate_documents(all_docs)[:self.k_retrieval]

        # Check if this is a page-specific query
        page_num = self._detect_page_query(query)

        if page_num is not None:
            if self.verbose:
                logger.info(f"[PAGE QUERY] Detected request for page {page_num}")
            # Use metadata filtering for page-specific queries
            try:
                docs = self.vector_store.similarity_search(
                    query,
                    k=self.k_retrieval * 3,  # Retrieve more to account for duplicates
                    filter={"page": page_num}
                )
                if docs:
                    docs = self._deduplicate_documents(docs)[:self.k_retrieval]
                    if self.verbose:
                        logger.info(f"[PAGE FILTER] Found {len(docs)} unique documents for page {page_num}")
                    return docs
                else:
                    if self.verbose:
                        logger.info(f"[PAGE FILTER] No documents found for page {page_num}, falling back to semantic search")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"[PAGE FILTER] Metadata filter failed: {e}, falling back to semantic search")

        # Check cache for search results
        if self.cache_manager and not bypass_cache:
            cached_results = self.cache_manager.get_search_results(query, self.k_retrieval)
            if cached_results:
                if self.verbose:
                    logger.info(f"[CACHE HIT] Retrieved {len(cached_results)} documents from cache")
                # Reconstruct documents from cache
                docs = []
                for doc_content, score in cached_results:
                    docs.append(Document(page_content=doc_content, metadata={"score": score, "source": "cached"}))
                return docs

        # Cache miss - perform actual search
        # When reranker is active, fetch more candidates for it to score
        if self.reranker:
            raw_k = max(self.reranker_candidates, self.k_retrieval * 3)
        else:
            raw_k = self.k_retrieval * 3
        raw_docs = self.vector_store.similarity_search(query, k=raw_k)

        if self.verbose:
            logger.info(f"[RETRIEVAL] Semantic search returned {len(raw_docs)} documents")

        # Supplement with keyword-based retrieval to catch chunks semantic search misses
        keyword_docs = self._keyword_search(query, k=self.k_retrieval)
        if keyword_docs:
            raw_docs = raw_docs + keyword_docs
            if self.verbose:
                logger.info(f"[RETRIEVAL] Keyword search added {len(keyword_docs)} supplemental documents")

        # Deduplicate results
        docs = self._deduplicate_documents(raw_docs)

        # Rerank with cross-encoder if available
        if self.reranker and len(docs) > 1:
            docs = self._rerank(query, docs, top_k=self.reranker_top_k)
        else:
            docs = docs[:self.k_retrieval]

        if self.verbose:
            logger.info(f"[RETRIEVAL] Returning {len(docs)} documents" +
                        (" (reranked)" if self.reranker else " (after deduplication)"))

        # Cache the deduplicated results
        if self.cache_manager and docs:
            results_to_cache = [(doc.page_content, 1.0) for doc in docs]
            self.cache_manager.cache_search_results(query, self.k_retrieval, results_to_cache)
            if self.verbose:
                logger.info(f"[CACHE MISS] Cached {len(docs)} unique documents")

        return docs

    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Public method to retrieve deduplicated documents.
        Use this instead of directly calling vector_store.similarity_search()

        Args:
            query: Search query
            k: Number of documents to retrieve (uses self.k_retrieval if None)

        Returns:
            List of unique, relevant documents
        """
        original_k = self.k_retrieval
        if k is not None:
            self.k_retrieval = k

        try:
            return self._retrieve_documents(query)
        finally:
            self.k_retrieval = original_k

    def _detect_summarization_query(self, query: str) -> bool:
        """Detect if query is asking for a summary or overview"""
        summarization_keywords = [
            'summarize', 'summary', 'summarise', 'overview', 'abstract',
            'main points', 'key points', 'key findings', 'key takeaways',
            'main contribution', 'main contributions', 'main idea',
            'tldr', 'recap', 'brief',
            'describe the paper', 'what does the paper say',
            'what is the paper about', 'what does this paper',
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in summarization_keywords)

    def _generate_response(self, query: str, context: str = "", require_citations: bool = True) -> str:
        """Generate response using Ollama with query-type-aware prompts"""

        if context:
            is_summarization = self._detect_summarization_query(query)

            if is_summarization:
                prompt = f"""You are summarizing a document based on the provided context.
Synthesize the information from ALL provided documents into a coherent, comprehensive summary.
Cover the main contributions, methodology, key findings, and conclusions.
Use the format [Document X] when referencing specific information.
Do NOT say you cannot summarize - work with the context provided.

Context:
{context}

Request: {query}

Summary (with citations):"""
            elif require_citations:
                prompt = f"""Answer the following question using the provided context.
IMPORTANT: You must cite specific information from the documents. Use the format [Document X] when referencing information.
Base your answer on the provided context. If the context contains relevant information, synthesize it into a clear answer.
Only say you cannot find the information if the context is truly unrelated to the question.

Context:
{context}

Question: {query}

Answer (with citations):"""
            else:
                prompt = f"""Answer the following question using the provided context. Be accurate and cite information from the context when possible.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question concisely and accurately:

Question: {query}

Answer:"""

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.verbose:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries}...")
                    else:
                        logger.info(f"Generating response with {self.model}...")

                start_time = time.time()

                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens,
                        'stop': ['Question:', 'Context:']
                    }
                )

                generation_time = time.time() - start_time

                answer = response['response'].strip()

                if self.verbose:
                    logger.info(f"Response generated in {generation_time:.1f}s, tokens: {response.get('eval_count', 'N/A')}")

                return answer

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Error generating response after {max_retries} attempts: {last_error}"
                    logger.error(error_msg)
                    return error_msg

    def query(self, question: str, use_retrieval: bool = True, bypass_cache: bool = False) -> str:
        """
        Query the RAG system

        Args:
            question: Question to ask
            use_retrieval: Whether to use document retrieval
            bypass_cache: Skip cache lookup (force fresh generation)

        Returns:
            Generated answer
        """

        if self.verbose:
            logger.info(f"OLLAMA RAG QUERY: {question[:60]}... | Retrieval: {use_retrieval} | Caching: {self.cache_manager is not None}")

        # Check cache for complete query response
        if self.cache_manager and not bypass_cache:
            cached = self.cache_manager.get_query_response(question)
            if cached:
                if self.verbose:
                    logger.info(f"[CACHE HIT] Returning cached response from {cached.get('cached_at', 'unknown')}")
                return cached["response"]

        context = ""
        docs = []
        is_summarization = self._detect_summarization_query(question)

        if use_retrieval:
            # Retrieve relevant documents (more for summarization)
            if is_summarization:
                docs = self._retrieve_documents(question)
                # For summarization, also retrieve with broader queries to get better coverage
                intro_docs = self._retrieve_documents("introduction background motivation")
                conclusion_docs = self._retrieve_documents("conclusion results findings contributions")
                # Merge and deduplicate
                all_docs = docs + intro_docs + conclusion_docs
                docs = self._deduplicate_documents(all_docs)[:self.k_retrieval * 2]
            else:
                docs = self._retrieve_documents(question)

            if docs:
                # Use all retrieved docs, with more content per doc for summarization
                max_chars_per_doc = 2000 if is_summarization else 1000
                context = "\n\n".join([
                    f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content[:max_chars_per_doc]}"
                    for i, doc in enumerate(docs)
                ])

                if self.verbose:
                    logger.info(f"Using {len(docs)} documents as context ({len(context)} chars)")
            else:
                if self.verbose:
                    logger.warning("No relevant documents found, using direct generation")
                use_retrieval = False

        # Generate response
        answer = self._generate_response(question, context)

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
        """
        Query with verification - shows exactly what context was used and compares
        retrieval vs no-retrieval answers to verify RAG is working.

        Returns:
            Dict with:
            - answer_with_retrieval: Answer using retrieved documents
            - answer_without_retrieval: Answer using only LLM knowledge
            - retrieved_context: The exact context passed to the LLM
            - retrieved_docs: List of retrieved document snippets
            - verification_notes: Analysis of whether retrieval made a difference
        """
        if self.verbose:
            logger.info(f"RAG VERIFICATION MODE: {question[:60]}...")

        # Step 1: Get answer WITH retrieval (bypass cache for fresh results)
        is_summarization = self._detect_summarization_query(question)

        # Bypass search cache for verification to ensure fresh retrieval
        docs = self._retrieve_documents(question, bypass_cache=True)

        # For summarization, also retrieve intro/conclusion chunks for broader coverage
        if is_summarization:
            intro_docs = self._retrieve_documents("introduction background motivation", bypass_cache=True)
            conclusion_docs = self._retrieve_documents("conclusion results findings contributions", bypass_cache=True)
            all_docs = docs + intro_docs + conclusion_docs
            docs = self._deduplicate_documents(all_docs)[:self.k_retrieval * 2]

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

        # Step 2: Get answer WITHOUT retrieval (pure LLM knowledge)
        answer_without_retrieval = self._generate_response(question, context="", require_citations=False)

        # Step 3: Analyze differences
        verification_notes = []

        if "cannot find" in answer_with_retrieval.lower() or "not found in" in answer_with_retrieval.lower():
            verification_notes.append("RAG answer indicates information not found in documents - may need better retrieval")

        if "[Document" in answer_with_retrieval:
            verification_notes.append("RAG answer includes citations - good sign retrieval is being used")

        # Check if answers are substantially different
        if len(answer_with_retrieval) > 0 and len(answer_without_retrieval) > 0:
            # Simple similarity check
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

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics"""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return None

    def clear_cache(self) -> None:
        """Clear all caches"""
        if self.cache_manager:
            self.cache_manager.clear_all()
            if self.verbose:
                logger.info("Cache cleared")

    def clear_vector_store(self) -> bool:
        """
        Clear the vector store completely, removing all persisted data.
        This should be called when starting a fresh session with new documents.

        Returns:
            True if successful, False otherwise
        """
        import shutil

        try:
            # Delete the collection if vector store exists
            if self.vector_store is not None:
                try:
                    # Get the underlying client and delete collection
                    self.vector_store.delete_collection()
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not delete collection: {e}")

                self.vector_store = None

            # Remove the persist directory to ensure clean state
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    logger.info(f"Deleted persist directory: {self.persist_directory}")

            # Clear document list
            self.documents = []

            # Clear caches
            self.clear_cache()

            if self.verbose:
                logger.info("Vector store cleared successfully")

            return True

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions"""

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
        """Get list of available Ollama models"""

        try:
            models = ollama.list()
            return [m.model for m in models.models]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model"""

        available_models = self.get_available_models()

        if new_model not in available_models:
            logger.error(f"Model {new_model} not available. Available: {available_models}")
            return False

        self.model = new_model

        if self.verbose:
            logger.info(f"Switched to model: {new_model}")

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""

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

def main():
    """Demo of Ollama RAG"""

    print("OLLAMA RAG DEMONSTRATION")
    print("="*50)

    try:
        # Initialize RAG system
        rag = OllamaRAG(
            model="llama2:7b", # Change to your preferred model
            verbose=True
        )

        # Show model info
        model_info = rag.get_model_info()
        print(f"\nModel Info: {json.dumps(model_info, indent=2)}")

        # Create sample documents
        documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
                metadata={"source": "ml_basics", "topic": "machine_learning"}
            ),
            Document(
                page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence applications.",
                metadata={"source": "python_intro", "topic": "programming"}
            ),
            Document(
                page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and informed responses.",
                metadata={"source": "rag_overview", "topic": "ai_techniques"}
            )
        ]

        # Add documents
        rag.add_documents(documents)

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How is Python used in AI?",
            "Explain RAG systems",
            "What programming languages are good for beginners?" # No direct context
        ]

        results = rag.batch_query(test_queries)

        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print('='*60)

        total_time = sum(r['processing_time'] for r in results)
        print(f"Total queries: {len(results)}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time: {total_time/len(results):.1f}s per query")

        print(f"\nAll {len(test_queries)} queries processed successfully!")

    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        print("\nSetup Instructions:")
        print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Download model: ollama pull llama2:7b")
        print("3. Install client: pip install ollama")

if __name__ == "__main__":
    main()
