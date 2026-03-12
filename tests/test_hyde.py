"""
Tests for HyDE (Hypothetical Document Embeddings) implementation.
All Ollama/LLM calls are mocked - no running Ollama required.
"""

import sys
import hashlib
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.experiments.hyde.ollama_hyde import HyDEResult, HyDERetrievalResult


# --- HyDEResult dataclass tests ---

class TestHyDEResult:
    def test_to_dict_basic(self):
        result = HyDEResult(
            query="What is ML?",
            hypothetical_document="ML is a field...",
            answer="Machine learning is...",
            retrieved_docs=[],
            hyde_retrieval_count=3,
            standard_retrieval_count=2,
            total_time=1.5,
            hyde_generation_time=0.5,
            retrieval_time=0.3,
            answer_generation_time=0.7,
        )
        d = result.to_dict()
        assert d["query"] == "What is ML?"
        assert d["answer"] == "Machine learning is..."
        assert d["hyde_retrieval_count"] == 3
        assert d["standard_retrieval_count"] == 2
        assert d["total_time"] == 1.5

    def test_to_dict_truncates_long_hypothetical(self):
        long_text = "x" * 600
        result = HyDEResult(
            query="q",
            hypothetical_document=long_text,
            answer="a",
            retrieved_docs=[],
            hyde_retrieval_count=0,
            standard_retrieval_count=0,
            total_time=0,
            hyde_generation_time=0,
            retrieval_time=0,
            answer_generation_time=0,
        )
        d = result.to_dict()
        assert d["hypothetical_document"].endswith("...")
        assert len(d["hypothetical_document"]) == 503  # 500 + "..."

    def test_to_dict_short_hypothetical_not_truncated(self):
        result = HyDEResult(
            query="q",
            hypothetical_document="short",
            answer="a",
            retrieved_docs=[],
            hyde_retrieval_count=0,
            standard_retrieval_count=0,
            total_time=0,
            hyde_generation_time=0,
            retrieval_time=0,
            answer_generation_time=0,
        )
        d = result.to_dict()
        assert d["hypothetical_document"] == "short"


class TestHyDERetrievalResult:
    def test_to_dict(self):
        from langchain.schema import Document
        docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        result = HyDERetrievalResult(
            query="test",
            hypothetical_document="hyp doc",
            documents=docs,
            retrieval_time=0.2,
            hyde_generation_time=0.3,
        )
        d = result.to_dict()
        assert d["document_count"] == 2
        assert d["retrieval_time"] == 0.2
        assert d["hypothetical_document"] == "hyp doc"

    def test_to_dict_truncates_long_hypothetical(self):
        result = HyDERetrievalResult(
            query="q",
            hypothetical_document="y" * 600,
            documents=[],
            retrieval_time=0,
            hyde_generation_time=0,
        )
        d = result.to_dict()
        assert d["hypothetical_document"].endswith("...")


# --- OllamaHyDE init and method tests (mocked) ---

@pytest.fixture
def mock_ollama():
    """Mock ollama module to avoid needing a running Ollama server."""
    mock_model = MagicMock()
    mock_model.model = "qwen2.5:14b"
    mock_list_result = MagicMock()
    mock_list_result.models = [mock_model]

    with patch('src.experiments.hyde.ollama_hyde.ollama') as mock_ol, \
         patch('src.experiments.hyde.ollama_hyde.OLLAMA_AVAILABLE', True), \
         patch('src.experiments.hyde.ollama_hyde.CONFIG_AVAILABLE', False):
        mock_ol.list.return_value = mock_list_result
        mock_ol.generate.return_value = {'response': 'mocked response'}
        yield mock_ol


@pytest.fixture
def hyde_instance(mock_ollama):
    """Create an OllamaHyDE instance with mocked dependencies."""
    with patch('src.core.embeddings.get_embeddings') as mock_emb:
        mock_emb.return_value = MagicMock()
        from src.experiments.hyde.ollama_hyde import OllamaHyDE
        instance = OllamaHyDE(model="qwen2.5:14b", verbose=False)
        return instance


class TestOllamaHyDEInit:
    def test_init_defaults(self, hyde_instance):
        assert hyde_instance.model == "qwen2.5:14b"
        assert hyde_instance.temperature == 0.7
        assert hyde_instance.answer_temperature == 0.3
        assert hyde_instance.k_retrieval == 5
        assert hyde_instance.vector_store is None
        assert hyde_instance.documents == []

    def test_init_custom_params(self, mock_ollama):
        with patch('src.core.embeddings.get_embeddings') as mock_emb:
            mock_emb.return_value = MagicMock()
            from src.experiments.hyde.ollama_hyde import OllamaHyDE
            instance = OllamaHyDE(
                model="qwen2.5:14b",
                temperature=0.5,
                answer_temperature=0.1,
                k_retrieval=10,
                max_tokens=500,
                verbose=False,
            )
            assert instance.temperature == 0.5
            assert instance.answer_temperature == 0.1
            assert instance.k_retrieval == 10
            assert instance.max_tokens == 500

    def test_init_raises_without_ollama(self):
        with patch('src.experiments.hyde.ollama_hyde.OLLAMA_AVAILABLE', False), \
             patch('src.experiments.hyde.ollama_hyde.CONFIG_AVAILABLE', False):
            from src.experiments.hyde.ollama_hyde import OllamaHyDE
            with pytest.raises(ImportError, match="ollama package not found"):
                OllamaHyDE()


class TestOllamaHyDEDeduplicate:
    def test_deduplicate_removes_duplicates(self, hyde_instance):
        from langchain.schema import Document
        docs = [
            Document(page_content="Hello world, this is a test document."),
            Document(page_content="Hello world, this is a test document."),
            Document(page_content="Different content entirely."),
        ]
        result = hyde_instance._deduplicate_docs(docs)
        assert len(result) == 2

    def test_deduplicate_empty_list(self, hyde_instance):
        assert hyde_instance._deduplicate_docs([]) == []

    def test_deduplicate_skips_empty_content(self, hyde_instance):
        from langchain.schema import Document
        docs = [
            Document(page_content="   "),
            Document(page_content="Valid content"),
        ]
        result = hyde_instance._deduplicate_docs(docs)
        assert len(result) == 1
        assert result[0].page_content == "Valid content"


class TestOllamaHyDECombineAndDeduplicate:
    def test_combines_and_deduplicates(self, hyde_instance):
        from langchain.schema import Document
        hyde_docs = [Document(page_content="shared content about machine learning", metadata={})]
        standard_docs = [
            Document(page_content="shared content about machine learning", metadata={}),
            Document(page_content="unique standard content about deep learning", metadata={}),
        ]
        result = hyde_instance._combine_and_deduplicate(hyde_docs, standard_docs)
        assert len(result) == 2
        assert result[0].metadata['retrieval_method'] == 'hyde'
        assert result[1].metadata['retrieval_method'] == 'standard'

    def test_hyde_docs_prioritized(self, hyde_instance):
        from langchain.schema import Document
        hyde_docs = [Document(page_content="first doc", metadata={})]
        standard_docs = [Document(page_content="second doc", metadata={})]
        result = hyde_instance._combine_and_deduplicate(hyde_docs, standard_docs)
        assert result[0].metadata['retrieval_method'] == 'hyde'


class TestOllamaHyDEGenerate:
    def test_generate_hypothetical_document(self, hyde_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': '  A hypothetical answer  '}
        result = hyde_instance._generate_hypothetical_document("What is ML?")
        assert result == "A hypothetical answer"
        mock_ollama.generate.assert_called_once()

    def test_generate_answer(self, hyde_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': '  The answer is 42  '}
        result = hyde_instance._generate_answer("question", "context", "hypothetical")
        assert result == "The answer is 42"


class TestOllamaHyDERetrieveWithHyDE:
    def test_retrieve_no_vector_store(self, hyde_instance):
        hyde_instance.vector_store = None
        hyde_docs, standard_docs = hyde_instance._retrieve_with_hypothetical("hyp", "query")
        assert hyde_docs == []
        assert standard_docs == []

    def test_retrieve_with_vector_store(self, hyde_instance):
        from langchain.schema import Document
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="result doc")
        ]
        hyde_instance.vector_store = mock_store
        hyde_docs, standard_docs = hyde_instance._retrieve_with_hypothetical("hyp doc", "query")
        assert len(hyde_docs) >= 1
        assert mock_store.similarity_search.call_count == 2
