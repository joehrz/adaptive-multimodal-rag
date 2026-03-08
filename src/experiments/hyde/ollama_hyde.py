"""
HyDE (Hypothetical Document Embeddings) Implementation with Ollama

Based on "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
https://arxiv.org/abs/2212.10496

HyDE improves retrieval by:
1. Generating a hypothetical document that would answer the query
2. Embedding the hypothetical document (not the query)
3. Using that embedding to retrieve similar real documents
4. Generating final answer from retrieved real documents

This bridges the semantic gap between short queries and longer documents.
"""

import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Import config system
try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    """Result from HyDE query"""
    query: str
    hypothetical_document: str
    answer: str
    retrieved_docs: List[Document]
    hyde_retrieval_count: int
    standard_retrieval_count: int
    total_time: float
    hyde_generation_time: float
    retrieval_time: float
    answer_generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "hypothetical_document": self.hypothetical_document[:500] + "..." if len(self.hypothetical_document) > 500 else self.hypothetical_document,
            "answer": self.answer,
            "hyde_retrieval_count": self.hyde_retrieval_count,
            "standard_retrieval_count": self.standard_retrieval_count,
            "total_time": self.total_time,
            "hyde_generation_time": self.hyde_generation_time,
            "retrieval_time": self.retrieval_time,
            "answer_generation_time": self.answer_generation_time
        }


@dataclass
class HyDERetrievalResult:
    """Result from HyDE retrieval-only operation (no answer generation)"""
    query: str
    hypothetical_document: str
    documents: List[Document]
    retrieval_time: float
    hyde_generation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "hypothetical_document": self.hypothetical_document[:500] + "..." if len(self.hypothetical_document) > 500 else self.hypothetical_document,
            "document_count": len(self.documents),
            "retrieval_time": self.retrieval_time,
            "hyde_generation_time": self.hyde_generation_time
        }


class OllamaHyDE:
    """
    HyDE (Hypothetical Document Embeddings) implementation using Ollama

    Key insight: Instead of embedding the query directly, we first generate
    a hypothetical document that would answer the query, then use THAT
    embedding to find similar real documents.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        answer_temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hypothetical_max_tokens: Optional[int] = None,
        k_retrieval: Optional[int] = None,
        verbose: Optional[bool] = None,
        persist_directory: Optional[str] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        dedup_min_chars: Optional[int] = None,
    ):
        """
        Initialize HyDE system

        Args:
            model: Ollama model for generation
            embedding_model: HuggingFace model for embeddings
            temperature: Temperature for hypothetical document generation
            answer_temperature: Temperature for final answer generation
            max_tokens: Max tokens for final answer
            hypothetical_max_tokens: Max tokens for hypothetical document
            k_retrieval: Number of documents to retrieve
            verbose: Enable verbose logging
            persist_directory: Directory for vector store persistence
            config: Optional Config object (uses global config if not provided)
            timeout: Timeout for LLM calls in seconds
            dedup_min_chars: Minimum characters for deduplication hash
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.temperature = temperature if temperature is not None else config.strategies.hyde.temperature
            self.answer_temperature = answer_temperature if answer_temperature is not None else config.strategies.hyde.answer_temperature
            self.max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
            self.hypothetical_max_tokens = hypothetical_max_tokens if hypothetical_max_tokens is not None else config.strategies.hyde.hypothetical_max_tokens
            self.k_retrieval = k_retrieval if k_retrieval is not None else config.documents.k_retrieval
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.persist_directory = persist_directory if persist_directory is not None else config.vector_db.hyde_persist_directory
            self.timeout = timeout if timeout is not None else config.llm.timeout
            self.dedup_min_chars = dedup_min_chars if dedup_min_chars is not None else config.documents.dedup_min_chars
            _embedding_model = embedding_model if embedding_model is not None else config.embeddings.model
            _embedding_device = config.embeddings.device
        else:
            # Fallback to hardcoded defaults if no config available
            self.model = model or "qwen2.5:14b"
            self.temperature = temperature if temperature is not None else 0.7
            self.answer_temperature = answer_temperature if answer_temperature is not None else 0.3
            self.max_tokens = max_tokens or 1000
            self.hypothetical_max_tokens = hypothetical_max_tokens or 300
            self.k_retrieval = k_retrieval or 5
            self.verbose = verbose if verbose is not None else True
            self.persist_directory = persist_directory or "./data/chroma_db_hyde"
            self.timeout = timeout or 120
            self.dedup_min_chars = dedup_min_chars or 500
            _embedding_model = embedding_model or "all-MiniLM-L6-v2"
            _embedding_device = "cpu"

        # Verify Ollama connection
        try:
            available_models = ollama.list()
            model_names = [m.model for m in available_models.models]
            if self.model not in model_names:
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")
            if self.verbose:
                logger.info(f"HyDE initialized with model: {self.model}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=_embedding_model,
            model_kwargs={'device': _embedding_device}
        )

        self.vector_store = None
        self.documents = []

        if self.verbose:
            logger.info(f"HyDE RAG initialized: model={self.model}, embedding={_embedding_model}, hyde_temp={self.temperature}, answer_temp={self.answer_temperature}")

    def add_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Add documents to the vector store"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        if not documents:
            if self.verbose:
                logger.warning("No documents provided to add_documents()")
            return

        self.documents.extend(documents)

        if self.verbose:
            logger.info(f"Processing {len(documents)} documents...")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(chunks)

        if self.verbose:
            logger.info(f"Added {len(chunks)} chunks to vector store")

    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        This is the core of HyDE - we create a fake but plausible answer
        to use for semantic retrieval.
        """
        prompt = f"""You are a knowledgeable assistant. Write a detailed, informative passage that would directly answer the following question.
Write as if this passage is from a textbook or authoritative source.
Do NOT say "I don't know" or ask questions - just write an informative passage.

Question: {query}

Informative passage:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.hypothetical_max_tokens,
            }
        )

        return response['response'].strip()

    def _retrieve_with_hypothetical(self, hypothetical_doc: str, query: str) -> Tuple[List[Document], List[Document]]:
        """
        Retrieve documents using both the hypothetical document embedding
        AND the original query embedding, then combine results.

        Returns:
            Tuple of (hyde_retrieved_docs, standard_retrieved_docs)
        """
        if not self.vector_store:
            return [], []

        # Retrieve MORE documents to account for duplicates (will be deduped in _combine_and_deduplicate)
        retrieval_multiplier = 3

        # Retrieve using hypothetical document embedding
        hyde_docs = self.vector_store.similarity_search(
            hypothetical_doc,
            k=self.k_retrieval * retrieval_multiplier
        )

        # Deduplicate hyde_docs
        hyde_docs = self._deduplicate_docs(hyde_docs)[:self.k_retrieval]

        # Also retrieve using standard query (for comparison/combination)
        standard_docs = self.vector_store.similarity_search(
            query,
            k=self.k_retrieval * retrieval_multiplier
        )

        # Deduplicate standard_docs
        standard_docs = self._deduplicate_docs(standard_docs)[:self.k_retrieval]

        return hyde_docs, standard_docs

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Deduplicate a list of documents based on content hash"""
        if not docs:
            return []

        seen_hashes = set()
        unique_docs = []

        for doc in docs:
            content = doc.page_content.strip()
            if not content:
                continue

            dedup_chars = min(self.dedup_min_chars, len(content))
            content_hash = hashlib.sha256(content[:dedup_chars].lower().encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def _combine_and_deduplicate(self, hyde_docs: List[Document], standard_docs: List[Document]) -> List[Document]:
        """Combine and deduplicate retrieved documents, prioritizing HyDE results"""
        seen_hashes = set()
        combined = []

        # Add HyDE docs first (higher priority)
        for doc in hyde_docs:
            # Use configurable minimum characters for robust deduplication
            dedup_chars = min(self.dedup_min_chars, len(doc.page_content))
            content_hash = hashlib.sha256(doc.page_content[:dedup_chars].encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                doc.metadata['retrieval_method'] = 'hyde'
                combined.append(doc)

        # Add standard docs that aren't duplicates
        for doc in standard_docs:
            dedup_chars = min(self.dedup_min_chars, len(doc.page_content))
            content_hash = hashlib.sha256(doc.page_content[:dedup_chars].encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                doc.metadata['retrieval_method'] = 'standard'
                combined.append(doc)

        return combined

    def _detect_summarization_query(self, query: str) -> bool:
        """Detect if query is asking for a summary or overview"""
        summarization_keywords = [
            'summarize', 'summary', 'summarise', 'overview', 'abstract',
            'main points', 'key points', 'tldr', 'recap', 'brief',
            'describe the paper', 'what does the paper say',
            'what is the paper about', 'what does this paper',
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in summarization_keywords)

    def _generate_answer(self, query: str, context: str, hypothetical: str) -> str:
        """Generate final answer using retrieved context with query-type-aware prompts"""

        if self._detect_summarization_query(query):
            prompt = f"""You are summarizing a document based on the provided context.
Synthesize the information from ALL provided documents into a coherent, comprehensive summary.
Cover the main contributions, methodology, key findings, and conclusions.
Use the format [Document X] when referencing specific information.
Do NOT say you cannot summarize - work with the context provided.

Context from retrieved documents:
{context}

Request: {query}

Summary (with citations):"""
        else:
            prompt = f"""Answer the following question using the provided context.
Be accurate and cite information from the context when relevant.
If the context doesn't contain the answer, say so.

Context from retrieved documents:
{context}

Question: {query}

Answer:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.answer_temperature,
                'num_predict': self.max_tokens,
            }
        )

        return response['response'].strip()

    def query(self, question: str) -> HyDEResult:
        """
        Query using HyDE methodology:
        1. Generate hypothetical document
        2. Use hypothetical embedding for retrieval
        3. Combine with standard retrieval
        4. Generate answer from retrieved docs
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"HyDE QUERY: {question[:60]}...")

        # Step 1: Generate hypothetical document
        if self.verbose:
            logger.info("[Step 1] Generating hypothetical document...")
        hyde_start = time.time()
        hypothetical_doc = self._generate_hypothetical_document(question)
        hyde_time = time.time() - hyde_start

        if self.verbose:
            logger.info(f"  Generated ({hyde_time:.1f}s): {hypothetical_doc[:100]}...")

        # Step 2: Retrieve using both hypothetical and standard
        if self.verbose:
            logger.info("[Step 2] Retrieving documents...")
        retrieval_start = time.time()
        hyde_docs, standard_docs = self._retrieve_with_hypothetical(hypothetical_doc, question)
        retrieval_time = time.time() - retrieval_start

        if self.verbose:
            logger.info(f"  HyDE retrieved: {len(hyde_docs)} docs, Standard retrieved: {len(standard_docs)} docs")

        # Step 3: Combine and deduplicate
        combined_docs = self._combine_and_deduplicate(hyde_docs, standard_docs)

        if self.verbose:
            logger.info(f"  Combined unique: {len(combined_docs)} docs")

        # Step 4: Build context and generate answer
        if self.verbose:
            logger.info("[Step 3] Generating answer...")

        context = ""
        if combined_docs:
            is_summarization = self._detect_summarization_query(question)
            max_chars = 2000 if is_summarization else 1000
            context = "\n\n".join([
                f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}, via {doc.metadata.get('retrieval_method', 'unknown')}): {doc.page_content[:max_chars]}"
                for i, doc in enumerate(combined_docs)
            ])
        else:
            if self.verbose:
                logger.warning("No documents retrieved - answer will be generated without context")

        answer_start = time.time()
        answer = self._generate_answer(question, context, hypothetical_doc)
        answer_time = time.time() - answer_start

        total_time = time.time() - start_time

        result = HyDEResult(
            query=question,
            hypothetical_document=hypothetical_doc,
            answer=answer,
            retrieved_docs=combined_docs,
            hyde_retrieval_count=len(hyde_docs),
            standard_retrieval_count=len(standard_docs),
            total_time=total_time,
            hyde_generation_time=hyde_time,
            retrieval_time=retrieval_time,
            answer_generation_time=answer_time
        )

        if self.verbose:
            logger.info(f"HyDE RESULT: total={total_time:.1f}s (hyde={hyde_time:.1f}s, retrieval={retrieval_time:.1f}s, answer={answer_time:.1f}s)")

        return result

    def retrieve_with_hyde(
        self,
        query: str,
        vector_store: Optional[Any] = None,
        top_k: Optional[int] = None
    ) -> HyDERetrievalResult:
        """
        Retrieve documents using HyDE methodology without generating an answer.

        This is useful for streaming scenarios where the caller wants to handle
        answer generation separately.

        Args:
            query: User query
            vector_store: Optional external vector store to use (uses internal if not provided)
            top_k: Number of documents to retrieve (uses self.k_retrieval if not provided)

        Returns:
            HyDERetrievalResult with hypothetical document and retrieved documents
        """
        start_time = time.time()
        k = top_k if top_k is not None else self.k_retrieval
        store = vector_store if vector_store is not None else self.vector_store

        if self.verbose:
            logger.info(f"HyDE RETRIEVAL: {query[:60]}...")

        # Step 1: Generate hypothetical document
        hyde_start = time.time()
        hypothetical_doc = self._generate_hypothetical_document(query)
        hyde_time = time.time() - hyde_start

        if self.verbose:
            logger.info(f"  Generated hypothetical ({hyde_time:.1f}s): {hypothetical_doc[:100]}...")

        # Step 2: Retrieve using hypothetical document embedding
        retrieval_start = time.time()
        documents = []
        retrieval_multiplier = 3  # Retrieve more to account for duplicates

        if store is not None:
            try:
                # Retrieve using the hypothetical document as the query
                raw_docs = store.similarity_search(hypothetical_doc, k=k * retrieval_multiplier)
                # Deduplicate
                documents = self._deduplicate_docs(raw_docs)[:k]

                if self.verbose:
                    logger.info(f"  Retrieved {len(documents)} unique documents using HyDE embedding (from {len(raw_docs)} raw)")
            except Exception as e:
                logger.warning(f"HyDE retrieval failed: {e}")
                # Fallback to standard query retrieval
                try:
                    raw_docs = store.similarity_search(query, k=k * retrieval_multiplier)
                    documents = self._deduplicate_docs(raw_docs)[:k]
                    if self.verbose:
                        logger.info(f"  Fallback: Retrieved {len(documents)} unique documents using standard query")
                except Exception as e2:
                    logger.error(f"Standard retrieval also failed: {e2}")
        else:
            if self.verbose:
                logger.warning("No vector store available for HyDE retrieval")

        retrieval_time = time.time() - retrieval_start

        result = HyDERetrievalResult(
            query=query,
            hypothetical_document=hypothetical_doc,
            documents=documents,
            retrieval_time=retrieval_time,
            hyde_generation_time=hyde_time
        )

        if self.verbose:
            total_time = time.time() - start_time
            logger.info(f"HyDE RETRIEVAL COMPLETE: {len(documents)} docs in {total_time:.1f}s")

        return result

    def query_compare(self, question: str) -> Dict[str, Any]:
        """
        Compare HyDE retrieval vs standard retrieval.
        Useful for evaluating HyDE effectiveness.
        """
        # Get HyDE result
        hyde_result = self.query(question)

        # Get standard retrieval result (without HyDE)
        if self.vector_store:
            standard_docs = self.vector_store.similarity_search(question, k=self.k_retrieval)
            standard_context = "\n\n".join([
                f"Document {i+1}: {doc.page_content[:800]}"
                for i, doc in enumerate(standard_docs[:5])
            ])

            standard_answer = self._generate_answer(question, standard_context, "")
        else:
            standard_docs = []
            standard_answer = "No documents available"

        return {
            "query": question,
            "hyde_answer": hyde_result.answer,
            "standard_answer": standard_answer,
            "hyde_docs_count": len(hyde_result.retrieved_docs),
            "standard_docs_count": len(standard_docs),
            "hypothetical_document": hyde_result.hypothetical_document,
            "hyde_time": hyde_result.total_time
        }

    def clear_vector_store(self) -> bool:
        """Clear the vector store"""
        import shutil
        import os

        try:
            if self.vector_store is not None:
                try:
                    self.vector_store.delete_collection()
                except Exception:
                    pass
                self.vector_store = None

            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            self.documents = []

            if self.verbose:
                logger.info("HyDE vector store cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False


def test_hyde():
    """Test HyDE functionality"""
    print("=" * 70)
    print("HyDE TEST")
    print("=" * 70)

    try:
        # Initialize
        print("\nInitializing HyDE...")
        hyde = OllamaHyDE(verbose=True)
        print("SUCCESS: HyDE initialized")

        # Create test documents
        documents = [
            Document(
                page_content="""Machine learning is a subset of artificial intelligence (AI) that enables
                computers to learn from data without being explicitly programmed. It uses algorithms
                to identify patterns and make predictions. Deep learning is a specialized form of
                machine learning that uses neural networks with multiple layers.""",
                metadata={"source": "ml_basics"}
            ),
            Document(
                page_content="""BERT (Bidirectional Encoder Representations from Transformers) is a
                pre-trained language model developed by Google. It uses masked language modeling
                and next sentence prediction for pre-training. BERT achieved state-of-the-art
                results on many NLP benchmarks including GLUE, SQuAD, and SWAG.""",
                metadata={"source": "bert_overview"}
            ),
            Document(
                page_content="""RAG (Retrieval-Augmented Generation) combines information retrieval
                with text generation. HyDE (Hypothetical Document Embeddings) improves RAG by
                generating a hypothetical answer first, then using that embedding to retrieve
                more relevant documents.""",
                metadata={"source": "rag_techniques"}
            )
        ]

        # Add documents
        hyde.add_documents(documents)

        # Test query
        query = "What is BERT and what tasks is it evaluated on?"
        print(f"\nQuery: {query}")

        result = hyde.query(query)

        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Hypothetical: {result.hypothetical_document[:200]}...")
        print(f"\nAnswer: {result.answer[:400]}...")
        print(f"\nHyDE retrieved: {result.hyde_retrieval_count} docs")
        print(f"Standard retrieved: {result.standard_retrieval_count} docs")
        print(f"Total time: {result.total_time:.1f}s")

        # Cleanup
        hyde.clear_vector_store()

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hyde()
