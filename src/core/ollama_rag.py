"""
Ollama-powered RAG implementation
Uses open-source LLMs (Llama, Mistral, etc.) running locally via Ollama
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
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

logger = logging.getLogger(__name__)

class OllamaRAG:
    """
    RAG implementation using Ollama for local open-source LLM generation
    Supports Llama 2, Mistral, CodeLlama, and other models
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        embedding_model: str = "all-MiniLM-L6-v2",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k_retrieval: int = 7,
        verbose: bool = True,
        cache_manager: Optional['RAGCacheManager'] = None,
        enable_caching: bool = True
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
        """

        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.k_retrieval = k_retrieval
        self.verbose = verbose

        # Test Ollama connection
        try:
            available_models = ollama.list()
            model_names = [m.model for m in available_models.models]

            if self.model not in model_names:
                if self.verbose:
                    print(f"WARNING: Model {self.model} not found locally")
                    print(f"Available models: {model_names}")
                    print(f"Download with: ollama pull {self.model}")
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")

            if self.verbose:
                print(f"SUCCESS: Ollama connected with model: {self.model}")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vector_store = None
        self.documents = []

        # Initialize caching
        self.enable_caching = enable_caching and CACHING_AVAILABLE
        if self.enable_caching:
            if cache_manager:
                self.cache_manager = cache_manager
            else:
                self.cache_manager = RAGCacheManager(enable_auto_cleanup=True)
            if self.verbose:
                print("Caching: Enabled")
        else:
            self.cache_manager = None
            if self.verbose:
                print("Caching: Disabled")

        if self.verbose:
            print(f"Ollama RAG initialized:")
            print(f"Model: {self.model}")
            print(f"Embedding: {embedding_model}")
            print(f"Chunk size: {chunk_size}")

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""

        self.documents.extend(documents)

        if self.verbose:
            print(f"Processing {len(documents)} documents...")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./data/chroma_db_ollama"
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(chunks)

        if self.verbose:
            print(f"Added {len(chunks)} chunks to vector store")

    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents with caching"""

        if not self.vector_store:
            if self.verbose:
                print("WARNING: No documents in vector store")
            return []

        # Check cache for search results
        if self.cache_manager:
            cached_results = self.cache_manager.get_search_results(query, self.k_retrieval)
            if cached_results:
                if self.verbose:
                    print(f"[CACHE HIT] Retrieved {len(cached_results)} documents from cache")
                # Reconstruct documents from cache (simplified - just content)
                docs = []
                for doc_content, score in cached_results:
                    docs.append(Document(page_content=doc_content, metadata={"score": score}))
                return docs

        # Cache miss - perform actual search
        docs = self.vector_store.similarity_search(query, k=self.k_retrieval)

        # Cache the results
        if self.cache_manager and docs:
            results_to_cache = [(doc.page_content, 1.0) for doc in docs]
            self.cache_manager.cache_search_results(query, self.k_retrieval, results_to_cache)
            if self.verbose:
                print(f"[CACHE MISS] Retrieved {len(docs)} documents and cached")
        elif self.verbose:
            print(f"Retrieved {len(docs)} relevant documents")

        return docs

    def _generate_response(self, query: str, context: str = "") -> str:
        """Generate response using Ollama"""

        if context:
            prompt = f"""Answer the following question using the provided context. Be accurate and cite information from the context when possible.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question concisely and accurately:

Question: {query}

Answer:"""

        try:
            if self.verbose:
                print(f"Generating response with {self.model}...")

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
                print(f"Response generated in {generation_time:.1f}s")
                print(f"Tokens: {response.get('eval_count', 'N/A')}")

            return answer

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            if self.verbose:
                print(f"ERROR: {error_msg}")
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
            print(f"\n{'='*60}")
            print(f"OLLAMA RAG QUERY")
            print(f"Question: {question}")
            print(f"Retrieval: {'Enabled' if use_retrieval else 'Disabled'}")
            print(f"Caching: {'Enabled' if self.cache_manager else 'Disabled'}")
            print('='*60)

        # Check cache for complete query response
        if self.cache_manager and not bypass_cache:
            cached = self.cache_manager.get_query_response(question)
            if cached:
                if self.verbose:
                    print(f"[CACHE HIT] Returning cached response")
                    print(f"Cached at: {cached.get('cached_at', 'unknown')}")
                    print(f"Strategy: {cached.get('strategy', 'unknown')}")
                return cached["response"]

        context = ""
        docs = []

        if use_retrieval:
            # Retrieve relevant documents
            docs = self._retrieve_documents(question)

            if docs:
                # Combine document contents as context
                context = "\n\n".join([
                    f"Document {i+1} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content[:1000]}..."
                    for i, doc in enumerate(docs[:5])
                ])

                if self.verbose:
                    print(f"Using {len(docs)} documents as context")
            else:
                if self.verbose:
                    print("No relevant documents found, using direct generation")

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
                print("[CACHED] Response cached for future queries")

        if self.verbose:
            print(f"\nAnswer: {answer}")

        return answer

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
                print("Cache cleared")

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions"""

        results = []

        for i, question in enumerate(questions, 1):
            if self.verbose:
                print(f"\n--- Query {i}/{len(questions)} ---")

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
            if self.verbose:
                print(f"Error getting models: {e}")
            return []

    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model"""

        available_models = self.get_available_models()

        if new_model not in available_models:
            if self.verbose:
                print(f"ERROR: Model {new_model} not available")
                print(f"Available: {available_models}")
                print(f"Download with: ollama pull {new_model}")
            return False

        self.model = new_model

        if self.verbose:
            print(f"SUCCESS: Switched to model: {new_model}")

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
