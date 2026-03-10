"""
Tests for core OllamaRAG implementation.
All Ollama/LLM calls are mocked - no running Ollama required.
"""

import sys
import hashlib
import time
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from langchain.schema import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama():
    """Mock ollama module to avoid needing a running Ollama server."""
    mock_model = MagicMock()
    mock_model.model = "qwen2.5:14b"
    mock_list_result = MagicMock()
    mock_list_result.models = [mock_model]

    with patch('src.core.ollama_rag.ollama') as mock_ol, \
         patch('src.core.ollama_rag.OLLAMA_AVAILABLE', True), \
         patch('src.core.ollama_rag.CONFIG_AVAILABLE', False), \
         patch('src.core.ollama_rag.CACHING_AVAILABLE', False):
        mock_ol.list.return_value = mock_list_result
        mock_ol.generate.return_value = {'response': 'mocked response', 'eval_count': 10}
        yield mock_ol


@pytest.fixture
def rag_instance(mock_ollama):
    """Create an OllamaRAG instance with mocked dependencies."""
    with patch('src.core.ollama_rag.HuggingFaceEmbeddings') as mock_emb:
        mock_emb.return_value = MagicMock()
        from src.core.ollama_rag import OllamaRAG
        instance = OllamaRAG(model="qwen2.5:14b", verbose=False)
        return instance


@pytest.fixture
def rag_instance_with_cache(mock_ollama):
    """Create an OllamaRAG instance with a mocked cache manager."""
    with patch('src.core.ollama_rag.HuggingFaceEmbeddings') as mock_emb:
        mock_emb.return_value = MagicMock()
        from src.core.ollama_rag import OllamaRAG
        instance = OllamaRAG(model="qwen2.5:14b", verbose=False)
        instance.cache_manager = MagicMock()
        instance.enable_caching = True
        return instance


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------

class TestOllamaRAGInit:
    def test_raises_import_error_when_ollama_unavailable(self):
        with patch('src.core.ollama_rag.OLLAMA_AVAILABLE', False), \
             patch('src.core.ollama_rag.CONFIG_AVAILABLE', False):
            from src.core.ollama_rag import OllamaRAG
            with pytest.raises(ImportError, match="ollama package not found"):
                OllamaRAG()

    def test_raises_connection_error_when_server_unreachable(self):
        with patch('src.core.ollama_rag.ollama') as mock_ol, \
             patch('src.core.ollama_rag.OLLAMA_AVAILABLE', True), \
             patch('src.core.ollama_rag.CONFIG_AVAILABLE', False), \
             patch('src.core.ollama_rag.HuggingFaceEmbeddings'):
            mock_ol.list.side_effect = Exception("Connection refused")
            from src.core.ollama_rag import OllamaRAG
            with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
                OllamaRAG(model="qwen2.5:14b")

    def test_raises_value_error_when_model_not_found(self):
        mock_model = MagicMock()
        mock_model.model = "llama2:7b"
        mock_list_result = MagicMock()
        mock_list_result.models = [mock_model]

        with patch('src.core.ollama_rag.ollama') as mock_ol, \
             patch('src.core.ollama_rag.OLLAMA_AVAILABLE', True), \
             patch('src.core.ollama_rag.CONFIG_AVAILABLE', False), \
             patch('src.core.ollama_rag.HuggingFaceEmbeddings'):
            mock_ol.list.return_value = mock_list_result
            from src.core.ollama_rag import OllamaRAG
            # Model "qwen2.5:14b" is not in the list (only "llama2:7b")
            # The ValueError is raised inside try block, then caught and re-raised as ConnectionError
            with pytest.raises(ConnectionError, match="Model qwen2.5:14b not available"):
                OllamaRAG(model="qwen2.5:14b")

    def test_initializes_with_config_defaults(self, rag_instance):
        # Without CONFIG_AVAILABLE, hardcoded defaults are used
        assert rag_instance.model == "qwen2.5:14b"
        assert rag_instance.temperature == 0.3
        assert rag_instance.max_tokens == 1000
        assert rag_instance.k_retrieval == 10
        assert rag_instance.verbose is False  # overridden by explicit param
        assert rag_instance.timeout == 120
        assert rag_instance.vector_store is None
        assert rag_instance.documents == []

    def test_initializes_with_explicit_parameters(self, mock_ollama):
        with patch('src.core.ollama_rag.HuggingFaceEmbeddings') as mock_emb:
            mock_emb.return_value = MagicMock()
            from src.core.ollama_rag import OllamaRAG
            instance = OllamaRAG(
                model="qwen2.5:14b",
                temperature=0.9,
                max_tokens=2000,
                k_retrieval=5,
                chunk_size=500,
                chunk_overlap=100,
                verbose=True,
                timeout=60,
            )
            assert instance.temperature == 0.9
            assert instance.max_tokens == 2000
            assert instance.k_retrieval == 5
            assert instance.verbose is True
            assert instance.timeout == 60

    def test_initializes_with_config_object(self, mock_ollama):
        """Explicit params override config values."""
        mock_config = MagicMock()
        mock_config.llm.model = "qwen2.5:14b"
        mock_config.llm.temperature = 0.5
        mock_config.llm.max_tokens = 800
        mock_config.llm.timeout = 90
        mock_config.documents.k_retrieval = 8
        mock_config.documents.chunk_size = 600
        mock_config.documents.chunk_overlap = 150
        mock_config.documents.dedup_min_chars = 300
        mock_config.embeddings.model = "all-MiniLM-L6-v2"
        mock_config.embeddings.device = "cpu"
        mock_config.vector_db.persist_directory = "/tmp/test_db"
        mock_config.logging.verbose = True
        mock_config.cache.enabled = False

        with patch('src.core.ollama_rag.HuggingFaceEmbeddings') as mock_emb:
            mock_emb.return_value = MagicMock()
            from src.core.ollama_rag import OllamaRAG
            instance = OllamaRAG(
                model="qwen2.5:14b",
                temperature=0.1,  # explicit override
                config=mock_config,
                verbose=False,
            )
            assert instance.temperature == 0.1  # explicit wins
            assert instance.max_tokens == 800  # from config
            assert instance.k_retrieval == 8  # from config
            assert instance.verbose is False  # explicit wins


# ---------------------------------------------------------------------------
# 2. Document management tests
# ---------------------------------------------------------------------------

class TestAddDocuments:
    def test_add_empty_list_warns_not_crashes(self, rag_instance):
        # Should not raise
        rag_instance.verbose = True
        rag_instance.add_documents([])
        assert rag_instance.vector_store is None

    def test_creates_vector_store_on_first_call(self, rag_instance):
        docs = [Document(page_content="Test content about machine learning.", metadata={"source": "test"})]
        with patch('src.core.ollama_rag.Chroma') as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            rag_instance.add_documents(docs)
            mock_chroma.from_documents.assert_called_once()
            assert rag_instance.vector_store is not None

    def test_adds_to_existing_vector_store(self, rag_instance):
        mock_store = MagicMock()
        rag_instance.vector_store = mock_store

        docs = [Document(page_content="Additional content for testing.", metadata={"source": "test2"})]
        rag_instance.add_documents(docs)
        mock_store.add_documents.assert_called_once()

    def test_documents_list_extended(self, rag_instance):
        docs = [Document(page_content="Content A"), Document(page_content="Content B")]
        with patch('src.core.ollama_rag.Chroma') as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            rag_instance.add_documents(docs)
            assert len(rag_instance.documents) == 2


class TestDeduplicateChunks:
    def test_removes_exact_duplicates(self, rag_instance):
        chunks = [
            Document(page_content="This is a duplicate chunk."),
            Document(page_content="This is a duplicate chunk."),
            Document(page_content="This is unique."),
        ]
        result = rag_instance._deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_keeps_unique_chunks(self, rag_instance):
        chunks = [
            Document(page_content="Chunk one about ML."),
            Document(page_content="Chunk two about NLP."),
            Document(page_content="Chunk three about CV."),
        ]
        result = rag_instance._deduplicate_chunks(chunks)
        assert len(result) == 3

    def test_empty_input(self, rag_instance):
        assert rag_instance._deduplicate_chunks([]) == []

    def test_skips_empty_content(self, rag_instance):
        chunks = [
            Document(page_content="   "),
            Document(page_content="Valid content here."),
        ]
        result = rag_instance._deduplicate_chunks(chunks)
        assert len(result) == 1
        assert result[0].page_content == "Valid content here."


class TestDeduplicateDocuments:
    def test_removes_duplicates_using_lowercase_hash(self, rag_instance):
        docs = [
            Document(page_content="Hello World document content."),
            Document(page_content="hello world document content."),  # same after lowercasing
        ]
        result = rag_instance._deduplicate_documents(docs)
        assert len(result) == 1

    def test_keeps_distinct_documents(self, rag_instance):
        docs = [
            Document(page_content="Document about machine learning algorithms."),
            Document(page_content="Document about natural language processing."),
        ]
        result = rag_instance._deduplicate_documents(docs)
        assert len(result) == 2

    def test_empty_input(self, rag_instance):
        assert rag_instance._deduplicate_documents([]) == []

    def test_skips_empty_content(self, rag_instance):
        docs = [
            Document(page_content=""),
            Document(page_content="  "),
            Document(page_content="Actual content."),
        ]
        result = rag_instance._deduplicate_documents(docs)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 3. Retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieveDocuments:
    def test_returns_empty_when_no_vector_store(self, rag_instance):
        rag_instance.vector_store = None
        result = rag_instance.retrieve_documents("What is ML?")
        assert result == []

    def test_delegates_to_vector_store_similarity_search(self, rag_instance):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Machine learning is a field of AI.")
        ]
        rag_instance.vector_store = mock_store
        result = rag_instance.retrieve_documents("What is ML?")
        assert len(result) >= 1
        mock_store.similarity_search.assert_called()


class TestKeywordSearch:
    def test_finds_documents_matching_query_terms(self, rag_instance):
        rag_instance.documents = [
            Document(page_content="Machine learning uses neural networks for prediction."),
            Document(page_content="Cooking recipes for dinner tonight."),
        ]
        result = rag_instance._keyword_search("machine learning prediction")
        assert len(result) >= 1
        assert "machine" in result[0].page_content.lower() or "learning" in result[0].page_content.lower()

    def test_returns_empty_for_short_words_only(self, rag_instance):
        rag_instance.documents = [
            Document(page_content="This is a test document."),
        ]
        # All words <= 3 chars: "is", "it", "ok"
        result = rag_instance._keyword_search("is it ok")
        assert result == []

    def test_returns_empty_when_no_documents(self, rag_instance):
        rag_instance.documents = []
        result = rag_instance._keyword_search("anything")
        assert result == []

    def test_respects_k_limit(self, rag_instance):
        rag_instance.documents = [
            Document(page_content=f"Document about machine learning number {i}")
            for i in range(20)
        ]
        result = rag_instance._keyword_search("machine learning", k=3)
        assert len(result) <= 3


class TestDetectMetadataQuery:
    def test_detects_title_question(self, rag_instance):
        assert rag_instance._detect_metadata_query("What is the title of this paper?") is True

    def test_detects_author_question(self, rag_instance):
        assert rag_instance._detect_metadata_query("Who are the authors of this paper?") is True

    def test_returns_false_for_normal_question(self, rag_instance):
        assert rag_instance._detect_metadata_query("How does gradient descent work?") is False

    def test_case_insensitive(self, rag_instance):
        assert rag_instance._detect_metadata_query("WHAT IS THE TITLE of this paper?") is True


class TestDetectSummarizationQuery:
    def test_detects_summarize_this_paper(self, rag_instance):
        assert rag_instance._detect_summarization_query("Summarize this paper") is True

    def test_detects_summary_keyword(self, rag_instance):
        assert rag_instance._detect_summarization_query("Give me a summary") is True

    def test_detects_overview_keyword(self, rag_instance):
        assert rag_instance._detect_summarization_query("Provide an overview of the paper") is True

    def test_returns_false_for_factual_questions(self, rag_instance):
        assert rag_instance._detect_summarization_query("What is the learning rate used?") is False

    def test_detects_what_is_paper_about(self, rag_instance):
        assert rag_instance._detect_summarization_query("What is the paper about?") is True


# ---------------------------------------------------------------------------
# 4. Query tests
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_with_retrieval_builds_context(self, rag_instance, mock_ollama):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="ML is a subset of AI.", metadata={"source": "test"})
        ]
        rag_instance.vector_store = mock_store

        result = rag_instance.query("What is ML?", use_retrieval=True)
        assert isinstance(result, str)
        assert len(result) > 0
        # ollama.generate should have been called with context in the prompt
        mock_ollama.generate.assert_called()
        call_args = mock_ollama.generate.call_args
        prompt = call_args[1]['prompt'] if 'prompt' in call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else ''
        # The prompt keyword arg should be present
        assert 'prompt' in call_args[1] or len(call_args[0]) > 1

    def test_query_without_retrieval_generates_directly(self, rag_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': 'Direct LLM answer', 'eval_count': 5}
        result = rag_instance.query("What is Python?", use_retrieval=False)
        assert result == "Direct LLM answer"
        mock_ollama.generate.assert_called_once()

    def test_query_summarization_retrieves_broader_context(self, rag_instance, mock_ollama):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Introduction of the paper.", metadata={"source": "test"})
        ]
        rag_instance.vector_store = mock_store

        result = rag_instance.query("Summarize this paper")
        # For summarization, _retrieve_documents is called 3 times (main + intro + conclusion)
        assert mock_store.similarity_search.call_count >= 3

    def test_query_uses_cache_when_available(self, rag_instance_with_cache, mock_ollama):
        rag_instance_with_cache.cache_manager.get_query_response.return_value = {
            "response": "Cached answer",
            "cached_at": "2026-03-10"
        }
        result = rag_instance_with_cache.query("What is ML?")
        assert result == "Cached answer"
        # Should NOT have called ollama.generate
        mock_ollama.generate.assert_not_called()

    def test_query_bypasses_cache_when_requested(self, rag_instance_with_cache, mock_ollama):
        rag_instance_with_cache.cache_manager.get_query_response.return_value = {
            "response": "Cached answer",
            "cached_at": "2026-03-10"
        }
        mock_ollama.generate.return_value = {'response': 'Fresh answer', 'eval_count': 5}

        result = rag_instance_with_cache.query("What is ML?", bypass_cache=True)
        assert result == "Fresh answer"
        # Cache lookup should NOT have been called
        rag_instance_with_cache.cache_manager.get_query_response.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Generation tests
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_calls_ollama_generate_with_correct_params(self, rag_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': 'Generated answer', 'eval_count': 10}
        result = rag_instance._generate_response("What is AI?", context="AI is intelligence by machines.")
        assert result == "Generated answer"
        mock_ollama.generate.assert_called_once()
        call_kwargs = mock_ollama.generate.call_args[1]
        assert call_kwargs['model'] == "qwen2.5:14b"
        assert 'temperature' in call_kwargs['options']
        assert call_kwargs['options']['temperature'] == 0.3

    def test_retries_on_failure(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = [
            Exception("Temporary error"),
            {'response': 'Success on retry', 'eval_count': 5},
        ]
        with patch('src.core.ollama_rag.time.sleep'):  # skip actual sleep
            result = rag_instance._generate_response("test query")
        assert result == "Success on retry"
        assert mock_ollama.generate.call_count == 2

    def test_returns_error_after_max_retries(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = Exception("Persistent error")
        with patch('src.core.ollama_rag.time.sleep'):  # skip actual sleep
            result = rag_instance._generate_response("test query")
        assert "Error generating response after 3 attempts" in result
        assert mock_ollama.generate.call_count == 3

    def test_no_context_generates_without_context_section(self, rag_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': 'No context answer', 'eval_count': 5}
        rag_instance._generate_response("What is Python?", context="")
        call_kwargs = mock_ollama.generate.call_args[1]
        prompt = call_kwargs['prompt']
        assert "Context:" not in prompt

    def test_summarization_prompt_used_for_summary_queries(self, rag_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': 'Summary response', 'eval_count': 10}
        rag_instance._generate_response("Summarize the paper", context="Some context here.")
        call_kwargs = mock_ollama.generate.call_args[1]
        prompt = call_kwargs['prompt']
        assert "summarizing" in prompt.lower()


# ---------------------------------------------------------------------------
# 6. Verification tests
# ---------------------------------------------------------------------------

class TestQueryWithVerification:
    def test_returns_both_rag_and_llm_answers(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = [
            {'response': 'RAG answer with [Document 1] citation', 'eval_count': 10},
            {'response': 'Pure LLM answer', 'eval_count': 8},
        ]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Relevant doc content.", metadata={"source": "test", "page": 1})
        ]
        rag_instance.vector_store = mock_store

        result = rag_instance.query_with_verification("What is ML?")

        assert "answer_with_retrieval" in result
        assert "answer_without_retrieval" in result
        assert result["answer_with_retrieval"] == "RAG answer with [Document 1] citation"
        assert result["answer_without_retrieval"] == "Pure LLM answer"

    def test_includes_retrieved_docs_info(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = [
            {'response': 'RAG answer', 'eval_count': 10},
            {'response': 'LLM answer', 'eval_count': 8},
        ]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Doc about neural networks.", metadata={"source": "paper.pdf", "page": 3})
        ]
        rag_instance.vector_store = mock_store

        result = rag_instance.query_with_verification("Explain neural networks")

        assert "retrieved_docs" in result
        assert "num_docs_retrieved" in result
        assert "context_length" in result
        assert "verification_notes" in result
        assert result["num_docs_retrieved"] >= 1
        assert len(result["retrieved_docs"]) >= 1
        assert result["retrieved_docs"][0]["source"] == "paper.pdf"

    def test_handles_no_vector_store(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = [
            {'response': 'No docs RAG answer', 'eval_count': 5},
            {'response': 'No docs LLM answer', 'eval_count': 5},
        ]
        rag_instance.vector_store = None

        result = rag_instance.query_with_verification("What is ML?")
        assert result["num_docs_retrieved"] == 0
        assert result["retrieved_docs"] == []
        assert result["context_length"] == 0

    def test_verification_notes_detect_citations(self, rag_instance, mock_ollama):
        mock_ollama.generate.side_effect = [
            {'response': 'Answer referencing [Document 1] here', 'eval_count': 10},
            {'response': 'Completely different LLM answer about something else', 'eval_count': 8},
        ]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Relevant content.", metadata={"source": "test"})
        ]
        rag_instance.vector_store = mock_store

        result = rag_instance.query_with_verification("Test question")
        notes = " ".join(result["verification_notes"])
        assert "citations" in notes.lower()


# ---------------------------------------------------------------------------
# 7. Utility method tests
# ---------------------------------------------------------------------------

class TestDetectPageQuery:
    def test_detects_page_number(self, rag_instance):
        assert rag_instance._detect_page_query("What is on page 5?") == 5

    def test_detects_pg_abbreviation(self, rag_instance):
        assert rag_instance._detect_page_query("Summarize pg 3") == 3

    def test_returns_none_for_no_page(self, rag_instance):
        assert rag_instance._detect_page_query("What is machine learning?") is None


class TestBatchQuery:
    def test_processes_multiple_questions(self, rag_instance, mock_ollama):
        mock_ollama.generate.return_value = {'response': 'batch answer', 'eval_count': 5}
        questions = ["Question 1?", "Question 2?"]
        results = rag_instance.batch_query(questions)
        assert len(results) == 2
        assert all('question' in r and 'answer' in r and 'processing_time' in r for r in results)


class TestGetAvailableModels:
    def test_returns_model_list(self, rag_instance, mock_ollama):
        mock_model1 = MagicMock()
        mock_model1.model = "qwen2.5:14b"
        mock_model2 = MagicMock()
        mock_model2.model = "llama2:7b"
        mock_list_result = MagicMock()
        mock_list_result.models = [mock_model1, mock_model2]
        mock_ollama.list.return_value = mock_list_result

        models = rag_instance.get_available_models()
        assert "qwen2.5:14b" in models
        assert "llama2:7b" in models

    def test_returns_empty_on_error(self, rag_instance, mock_ollama):
        mock_ollama.list.side_effect = Exception("connection error")
        models = rag_instance.get_available_models()
        assert models == []


class TestSwitchModel:
    def test_switches_to_available_model(self, rag_instance, mock_ollama):
        mock_model = MagicMock()
        mock_model.model = "llama2:7b"
        mock_list_result = MagicMock()
        mock_list_result.models = [mock_model]
        mock_ollama.list.return_value = mock_list_result

        result = rag_instance.switch_model("llama2:7b")
        assert result is True
        assert rag_instance.model == "llama2:7b"

    def test_fails_for_unavailable_model(self, rag_instance, mock_ollama):
        mock_list_result = MagicMock()
        mock_list_result.models = []
        mock_ollama.list.return_value = mock_list_result

        result = rag_instance.switch_model("nonexistent:model")
        assert result is False


class TestClearVectorStore:
    def test_clears_vector_store(self, rag_instance):
        mock_store = MagicMock()
        rag_instance.vector_store = mock_store
        rag_instance.documents = [Document(page_content="test")]
        rag_instance.persist_directory = "/tmp/nonexistent_test_dir"

        result = rag_instance.clear_vector_store()
        assert result is True
        assert rag_instance.vector_store is None
        assert rag_instance.documents == []
        mock_store.delete_collection.assert_called_once()
