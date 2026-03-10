"""
Configuration management for Adaptive Multimodal RAG

Loads settings from config.yaml with environment variable overrides.
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails"""
    pass


@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str = "qwen2.5:14b"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: int = 120  # Default timeout in seconds for LLM calls

    def __post_init__(self):
        if not 0 <= self.temperature <= 2:
            raise ConfigValidationError(f"LLM temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ConfigValidationError(f"LLM max_tokens must be positive, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ConfigValidationError(f"LLM timeout must be positive, got {self.timeout}")


@dataclass
class EmbeddingsConfig:
    """Embedding model configuration"""
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


@dataclass
class RerankerConfig:
    """Cross-encoder reranker configuration"""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    top_k: int = 10
    candidates: int = 30

    def __post_init__(self):
        if self.top_k <= 0:
            raise ConfigValidationError(f"reranker top_k must be positive, got {self.top_k}")
        if self.candidates <= 0:
            raise ConfigValidationError(f"reranker candidates must be positive, got {self.candidates}")
        if self.top_k > self.candidates:
            raise ConfigValidationError(f"reranker top_k ({self.top_k}) must be <= candidates ({self.candidates})")


@dataclass
class DocumentsConfig:
    """Document processing configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k_retrieval: int = 10
    dedup_min_chars: int = 500  # Minimum characters for deduplication hash

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ConfigValidationError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ConfigValidationError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigValidationError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        if self.k_retrieval <= 0:
            raise ConfigValidationError(f"k_retrieval must be positive, got {self.k_retrieval}")


@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    persist_directory: str = "./data/chroma_db_ollama"
    hyde_persist_directory: str = "./data/chroma_db_hyde"
    self_rag_persist_directory: str = "./data/chroma_db_self_rag"


@dataclass
class OllamaConfig:
    """Ollama server configuration"""
    url: str = "http://localhost:11434"
    timeout: int = 120  # Default request timeout

    def __post_init__(self):
        if not self.url:
            raise ConfigValidationError("Ollama URL cannot be empty")
        if self.timeout <= 0:
            raise ConfigValidationError(f"Ollama timeout must be positive, got {self.timeout}")


@dataclass
class HyDEConfig:
    """HyDE strategy configuration"""
    temperature: float = 0.7
    hypothetical_max_tokens: int = 300
    answer_temperature: float = 0.3

    def __post_init__(self):
        if not 0 <= self.temperature <= 2:
            raise ConfigValidationError(f"HyDE temperature must be between 0 and 2, got {self.temperature}")
        if self.hypothetical_max_tokens <= 0:
            raise ConfigValidationError(f"hypothetical_max_tokens must be positive, got {self.hypothetical_max_tokens}")
        if not 0 <= self.answer_temperature <= 2:
            raise ConfigValidationError(f"answer_temperature must be between 0 and 2, got {self.answer_temperature}")


@dataclass
class SelfRAGConfig:
    """Self-RAG configuration"""
    max_regenerations: int = 2
    quality_threshold: float = 0.5
    reflection_temperature: float = 0.1  # Low temp for consistent reflection

    def __post_init__(self):
        if self.max_regenerations < 0:
            raise ConfigValidationError(f"max_regenerations must be non-negative, got {self.max_regenerations}")
        if not 0 <= self.quality_threshold <= 1:
            raise ConfigValidationError(f"quality_threshold must be between 0 and 1, got {self.quality_threshold}")
        if not 0 <= self.reflection_temperature <= 2:
            raise ConfigValidationError(f"reflection_temperature must be between 0 and 2, got {self.reflection_temperature}")


@dataclass
class GraphRAGConfig:
    """GraphRAG configuration"""
    max_hops: int = 3
    timeout: int = 60
    max_entities_per_doc: int = 7
    max_relationships_per_doc: int = 5

    def __post_init__(self):
        if self.max_hops <= 0:
            raise ConfigValidationError(f"max_hops must be positive, got {self.max_hops}")
        if self.timeout <= 0:
            raise ConfigValidationError(f"timeout must be positive, got {self.timeout}")
        if self.max_entities_per_doc <= 0:
            raise ConfigValidationError(f"max_entities_per_doc must be positive, got {self.max_entities_per_doc}")
        if self.max_relationships_per_doc <= 0:
            raise ConfigValidationError(f"max_relationships_per_doc must be positive, got {self.max_relationships_per_doc}")


@dataclass
class StrategiesConfig:
    """Strategy routing configuration"""
    simple_max: int = 3
    medium_max: int = 7
    complex_min: int = 8
    graphrag_min: int = 6
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    self_rag: SelfRAGConfig = field(default_factory=SelfRAGConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)

    def __post_init__(self):
        if self.simple_max < 0 or self.simple_max > 10:
            raise ConfigValidationError(f"simple_max must be between 0 and 10, got {self.simple_max}")
        if self.medium_max < 0 or self.medium_max > 10:
            raise ConfigValidationError(f"medium_max must be between 0 and 10, got {self.medium_max}")
        if self.complex_min < 0 or self.complex_min > 10:
            raise ConfigValidationError(f"complex_min must be between 0 and 10, got {self.complex_min}")
        if self.simple_max >= self.medium_max:
            raise ConfigValidationError(f"simple_max ({self.simple_max}) must be less than medium_max ({self.medium_max})")
        if self.medium_max >= self.complex_min:
            raise ConfigValidationError(f"medium_max ({self.medium_max}) must be less than complex_min ({self.complex_min})")


@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    auto_cleanup: bool = True
    ttl_seconds: int = 3600
    query_cache_capacity: int = 500
    vector_cache_capacity: int = 1000

    def __post_init__(self):
        if self.ttl_seconds <= 0:
            raise ConfigValidationError(f"ttl_seconds must be positive, got {self.ttl_seconds}")
        if self.query_cache_capacity <= 0:
            raise ConfigValidationError(f"query_cache_capacity must be positive, got {self.query_cache_capacity}")
        if self.vector_cache_capacity <= 0:
            raise ConfigValidationError(f"vector_cache_capacity must be positive, got {self.vector_cache_capacity}")


@dataclass
class UIConfig:
    """UI configuration"""
    page_title: str = "Adaptive RAG System"
    default_strategy: str = "adaptive"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    verbose: bool = False


@dataclass
class Config:
    """Main configuration container"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _get_nested_value(d: Dict, keys: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation"""
    keys_list = keys.split('.')
    value = d
    for key in keys_list:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def _apply_env_overrides(config_dict: Dict) -> Dict:
    """Apply environment variable overrides to config"""
    # Environment variable mappings (ENV_VAR -> config.path)
    env_mappings = {
        'RAG_MODEL': 'llm.model',
        'RAG_TEMPERATURE': 'llm.temperature',
        'RAG_MAX_TOKENS': 'llm.max_tokens',
        'RAG_EMBEDDING_MODEL': 'embeddings.model',
        'RAG_CHUNK_SIZE': 'documents.chunk_size',
        'RAG_K_RETRIEVAL': 'documents.k_retrieval',
        'RAG_VECTOR_DB_PATH': 'vector_db.persist_directory',
        'RAG_CACHE_ENABLED': 'cache.enabled',
        'RAG_LOG_LEVEL': 'logging.level',
        'RAG_VERBOSE': 'logging.verbose',
        'RAG_OLLAMA_URL': 'ollama.url',
        'RAG_OLLAMA_TIMEOUT': 'ollama.timeout',
    }

    for env_var, config_path in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Parse the value based on expected type
            keys = config_path.split('.')
            current = config_dict

            # Navigate to parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value with type conversion
            final_key = keys[-1]
            if env_value.lower() in ('true', 'false'):
                current[final_key] = env_value.lower() == 'true'
            elif env_value.isdigit():
                current[final_key] = int(env_value)
            elif env_value.replace('.', '').isdigit():
                current[final_key] = float(env_value)
            else:
                current[final_key] = env_value

    return config_dict


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables
    2. config.yaml values
    3. Default values

    Args:
        config_path: Path to config file. If None, searches for config.yaml in project root.

    Returns:
        Config object with all settings
    """
    # Find config file
    if config_path is None:
        # Search in common locations
        search_paths = [
            Path("config.yaml"),
            Path("../config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    # Load YAML if found
    config_dict: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Build config objects
    config = Config(
        llm=LLMConfig(
            model=_get_nested_value(config_dict, 'llm.model', 'qwen2.5:14b'),
            temperature=float(_get_nested_value(config_dict, 'llm.temperature', 0.3)),
            max_tokens=int(_get_nested_value(config_dict, 'llm.max_tokens', 1000)),
            timeout=int(_get_nested_value(config_dict, 'llm.timeout', 120)),
        ),
        embeddings=EmbeddingsConfig(
            model=_get_nested_value(config_dict, 'embeddings.model', 'all-MiniLM-L6-v2'),
            device=_get_nested_value(config_dict, 'embeddings.device', 'cpu'),
        ),
        reranker=RerankerConfig(
            enabled=bool(_get_nested_value(config_dict, 'reranker.enabled', True)),
            model=_get_nested_value(config_dict, 'reranker.model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
            device=_get_nested_value(config_dict, 'reranker.device', 'cpu'),
            top_k=int(_get_nested_value(config_dict, 'reranker.top_k', 10)),
            candidates=int(_get_nested_value(config_dict, 'reranker.candidates', 30)),
        ),
        documents=DocumentsConfig(
            chunk_size=int(_get_nested_value(config_dict, 'documents.chunk_size', 1000)),
            chunk_overlap=int(_get_nested_value(config_dict, 'documents.chunk_overlap', 200)),
            k_retrieval=int(_get_nested_value(config_dict, 'documents.k_retrieval', 10)),
            dedup_min_chars=int(_get_nested_value(config_dict, 'documents.dedup_min_chars', 500)),
        ),
        vector_db=VectorDBConfig(
            persist_directory=_get_nested_value(config_dict, 'vector_db.persist_directory', './data/chroma_db_ollama'),
            hyde_persist_directory=_get_nested_value(config_dict, 'vector_db.hyde_persist_directory', './data/chroma_db_hyde'),
            self_rag_persist_directory=_get_nested_value(config_dict, 'vector_db.self_rag_persist_directory', './data/chroma_db_self_rag'),
        ),
        ollama=OllamaConfig(
            url=_get_nested_value(config_dict, 'ollama.url', 'http://localhost:11434'),
            timeout=int(_get_nested_value(config_dict, 'ollama.timeout', 120)),
        ),
        strategies=StrategiesConfig(
            simple_max=int(_get_nested_value(config_dict, 'strategies.simple_max', 3)),
            medium_max=int(_get_nested_value(config_dict, 'strategies.medium_max', 7)),
            complex_min=int(_get_nested_value(config_dict, 'strategies.complex_min', 8)),
            graphrag_min=int(_get_nested_value(config_dict, 'strategies.graphrag_min', 6)),
            hyde=HyDEConfig(
                temperature=float(_get_nested_value(config_dict, 'strategies.hyde.temperature', 0.7)),
                hypothetical_max_tokens=int(_get_nested_value(config_dict, 'strategies.hyde.hypothetical_max_tokens', 300)),
                answer_temperature=float(_get_nested_value(config_dict, 'strategies.hyde.answer_temperature', 0.3)),
            ),
            self_rag=SelfRAGConfig(
                max_regenerations=int(_get_nested_value(config_dict, 'strategies.self_rag.max_regenerations', 2)),
                quality_threshold=float(_get_nested_value(config_dict, 'strategies.self_rag.quality_threshold', 0.5)),
                reflection_temperature=float(_get_nested_value(config_dict, 'strategies.self_rag.reflection_temperature', 0.1)),
            ),
            graphrag=GraphRAGConfig(
                max_hops=int(_get_nested_value(config_dict, 'strategies.graphrag.max_hops', 3)),
                timeout=int(_get_nested_value(config_dict, 'strategies.graphrag.timeout', 60)),
                max_entities_per_doc=int(_get_nested_value(config_dict, 'strategies.graphrag.max_entities_per_doc', 7)),
                max_relationships_per_doc=int(_get_nested_value(config_dict, 'strategies.graphrag.max_relationships_per_doc', 5)),
            ),
        ),
        cache=CacheConfig(
            enabled=bool(_get_nested_value(config_dict, 'cache.enabled', True)),
            auto_cleanup=bool(_get_nested_value(config_dict, 'cache.auto_cleanup', True)),
            ttl_seconds=int(_get_nested_value(config_dict, 'cache.ttl_seconds', 3600)),
            query_cache_capacity=int(_get_nested_value(config_dict, 'cache.query_cache_capacity', 500)),
            vector_cache_capacity=int(_get_nested_value(config_dict, 'cache.vector_cache_capacity', 1000)),
        ),
        ui=UIConfig(
            page_title=_get_nested_value(config_dict, 'ui.page_title', 'Adaptive RAG System'),
            default_strategy=_get_nested_value(config_dict, 'ui.default_strategy', 'adaptive'),
        ),
        logging=LoggingConfig(
            level=_get_nested_value(config_dict, 'logging.level', 'INFO'),
            verbose=bool(_get_nested_value(config_dict, 'logging.verbose', False)),
        ),
    )

    logger.debug(f"Configuration loaded: model={config.llm.model}, embeddings={config.embeddings.model}")
    return config


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file"""
    global _config
    _config = load_config(config_path)
    return _config


# Convenience function
def get_model() -> str:
    """Get configured LLM model name"""
    return get_config().llm.model


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Chunk Size: {config.documents.chunk_size}")
    print(f"  K Retrieval: {config.documents.k_retrieval}")
    print(f"  Cache Enabled: {config.cache.enabled}")
