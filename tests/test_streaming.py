"""
Tests for Streaming RAG implementation.
All network/Ollama calls are mocked - no running Ollama required.
"""

import sys
import time
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.experiments.streaming.ollama_streaming_rag import (
    StreamingStage,
    StreamChunk,
    StreamingProgress,
    OllamaStreamingRAG,
    StreamingConsoleDisplay,
)


# --- Enum and dataclass tests ---

class TestStreamingStage:
    def test_all_stages_exist(self):
        assert StreamingStage.ANALYZING_QUERY.value == "analyzing_query"
        assert StreamingStage.RETRIEVING_DOCS.value == "retrieving_docs"
        assert StreamingStage.GENERATING_HYPOTHETICALS.value == "generating_hypotheticals"
        assert StreamingStage.RUNNING_REFLECTION.value == "running_reflection"
        assert StreamingStage.GENERATING_ANSWER.value == "generating_answer"
        assert StreamingStage.COMPLETE.value == "complete"

    def test_stage_count(self):
        assert len(StreamingStage) == 6


class TestStreamChunk:
    def test_creation(self):
        chunk = StreamChunk(
            stage=StreamingStage.GENERATING_ANSWER,
            content="hello",
            metadata={"token_count": 1},
            timestamp=1000.0,
        )
        assert chunk.stage == StreamingStage.GENERATING_ANSWER
        assert chunk.content == "hello"
        assert chunk.metadata["token_count"] == 1
        assert chunk.timestamp == 1000.0

    def test_stage_can_be_reassigned(self):
        chunk = StreamChunk(
            stage=StreamingStage.GENERATING_ANSWER,
            content="x",
            metadata={},
            timestamp=0,
        )
        chunk.stage = StreamingStage.GENERATING_HYPOTHETICALS
        assert chunk.stage == StreamingStage.GENERATING_HYPOTHETICALS


class TestStreamingProgress:
    def test_creation(self):
        progress = StreamingProgress(
            stage=StreamingStage.RETRIEVING_DOCS,
            progress=0.5,
            message="Retrieving...",
            timestamp=123.0,
        )
        assert progress.stage == StreamingStage.RETRIEVING_DOCS
        assert progress.progress == 0.5
        assert progress.message == "Retrieving..."


# --- OllamaStreamingRAG init tests ---

class TestOllamaStreamingRAGInit:
    def test_init_defaults_no_config(self):
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(model="test-model", verbose=False)
            assert rag.model == "test-model"
            assert rag.ollama_url == "http://localhost:11434"
            assert rag.verbose is False
            assert rag.timeout == 300
            assert rag._hyde_engine is None

    def test_init_with_custom_params(self):
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(
                model="custom",
                ollama_url="http://custom:1234",
                verbose=True,
                timeout=60,
            )
            assert rag.model == "custom"
            assert rag.ollama_url == "http://custom:1234"
            assert rag.timeout == 60

    def test_init_with_config(self):
        mock_config = MagicMock()
        mock_config.llm.model = "config-model"
        mock_config.ollama.url = "http://config:5555"
        mock_config.logging.verbose = True
        mock_config.ollama.timeout = 200

        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', True), \
             patch('src.experiments.streaming.ollama_streaming_rag.get_config', return_value=mock_config):
            rag = OllamaStreamingRAG()
            assert rag.model == "config-model"
            assert rag.ollama_url == "http://config:5555"
            assert rag.timeout == 200


# --- Streaming generation tests (mocked HTTP) ---

class TestStreamGenerate:
    def test_stream_generate_yields_chunks(self):
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(model="test", verbose=False)

        import json
        lines = [
            json.dumps({"response": "Hello"}).encode(),
            json.dumps({"response": " world"}).encode(),
            json.dumps({"response": "!", "done": True}).encode(),
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_response.raise_for_status = MagicMock()

        with patch('src.experiments.streaming.ollama_streaming_rag.requests.post', return_value=mock_response):
            chunks = list(rag.stream_generate("test prompt"))

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        assert all(c.stage == StreamingStage.GENERATING_ANSWER for c in chunks)

    def test_stream_generate_calls_token_callback(self):
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(model="test", verbose=False)

        import json
        lines = [
            json.dumps({"response": "tok", "done": True}).encode(),
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_response.raise_for_status = MagicMock()

        tokens = []
        with patch('src.experiments.streaming.ollama_streaming_rag.requests.post', return_value=mock_response):
            list(rag.stream_generate("prompt", on_token=lambda t: tokens.append(t)))

        assert tokens == ["tok"]

    def test_stream_generate_handles_request_error(self):
        import requests as req
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(model="test", verbose=False)

        with patch('src.experiments.streaming.ollama_streaming_rag.requests.post',
                   side_effect=req.RequestException("connection refused")):
            chunks = list(rag.stream_generate("prompt"))

        assert len(chunks) == 1
        assert "error" in chunks[0].metadata


class TestStreamRAGQuery:
    def test_stream_rag_query_builds_context(self):
        from langchain.schema import Document
        with patch('src.experiments.streaming.ollama_streaming_rag.CONFIG_AVAILABLE', False):
            rag = OllamaStreamingRAG(model="test", verbose=False)

        import json
        lines = [json.dumps({"response": "answer", "done": True}).encode()]
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_response.raise_for_status = MagicMock()

        docs = [Document(page_content="test content", metadata={"source": "test"})]

        with patch('src.experiments.streaming.ollama_streaming_rag.requests.post', return_value=mock_response) as mock_post:
            chunks = list(rag.stream_rag_query("question?", docs))

        # Verify the prompt includes document content
        call_args = mock_post.call_args
        prompt = call_args[1]["json"]["prompt"] if "json" in call_args[1] else call_args[0][1]["prompt"]
        assert "test content" in prompt


# --- StreamingConsoleDisplay tests ---

class TestStreamingConsoleDisplay:
    def test_on_token_prints(self, capsys):
        display = StreamingConsoleDisplay(show_tokens=True)
        display.on_token("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_on_token_silent_when_disabled(self, capsys):
        display = StreamingConsoleDisplay(show_tokens=False)
        display.on_token("hello")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_on_progress_shows_stage_change(self, capsys):
        display = StreamingConsoleDisplay(show_progress=True)
        progress = StreamingProgress(
            stage=StreamingStage.GENERATING_ANSWER,
            progress=0.5,
            message="Generating...",
            timestamp=time.time(),
        )
        display.on_progress(progress)
        captured = capsys.readouterr()
        assert "Generating..." in captured.out

    def test_on_progress_skips_same_stage(self, capsys):
        display = StreamingConsoleDisplay(show_progress=True)
        progress = StreamingProgress(
            stage=StreamingStage.GENERATING_ANSWER,
            progress=0.5,
            message="Generating...",
            timestamp=time.time(),
        )
        display.on_progress(progress)
        capsys.readouterr()  # clear

        display.on_progress(progress)  # same stage
        captured = capsys.readouterr()
        assert captured.out == ""
