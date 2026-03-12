"""
Embedding backend factory for Adaptive Multimodal RAG

Supports HuggingFace (default) and Ollama embedding backends.
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import EmbeddingsConfig

logger = logging.getLogger(__name__)


def get_embeddings(
    config: Optional['EmbeddingsConfig'] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    ollama_url: Optional[str] = None,
):
    """
    Create an embeddings instance based on configuration.

    Args:
        config: Optional EmbeddingsConfig object
        backend: "huggingface" or "ollama" (overrides config)
        model: Model name (overrides config)
        device: Device for HuggingFace models (overrides config)
        ollama_url: Ollama server URL (overrides config)

    Returns:
        An embeddings instance (HuggingFaceEmbeddings or OllamaEmbeddings)
    """
    # Resolve values: explicit args > config > defaults
    _backend = backend or (config.backend if config else None) or "huggingface"
    _device = device or (config.device if config else None) or "cpu"

    if _backend == "ollama":
        _model = model or (config.model if config else None) or "nomic-embed-text"
        _url = ollama_url or "http://localhost:11434"
        logger.info(f"Using Ollama embeddings: model={_model}, url={_url}")
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=_model, base_url=_url)
    else:
        _model = model or (config.model if config else None) or "all-MiniLM-L6-v2"
        logger.info(f"Using HuggingFace embeddings: model={_model}, device={_device}")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=_model,
            model_kwargs={'device': _device},
        )
