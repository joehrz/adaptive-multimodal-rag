"""
Test script for Config validation with __post_init__ checks
"""

import sys
sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.core.config import (
    ConfigValidationError,
    LLMConfig,
    DocumentsConfig,
    HyDEConfig,
    SelfRAGConfig,
    GraphRAGConfig,
    StrategiesConfig,
    CacheConfig,
    OllamaConfig,
)


def test_llm_config_validation():
    """Test LLMConfig validation"""
    print("Testing LLMConfig validation...")

    # Valid config
    config = LLMConfig(model="test", temperature=0.5, max_tokens=100, timeout=60)
    print("  Valid config: PASS")

    # Invalid temperature (too high)
    try:
        LLMConfig(temperature=3.0)
        assert False, "Should have raised ConfigValidationError for temperature=3.0"
    except ConfigValidationError:
        pass

    # Invalid max_tokens
    try:
        LLMConfig(max_tokens=0)
        assert False, "Should have raised ConfigValidationError for max_tokens=0"
    except ConfigValidationError:
        pass


def test_documents_config_validation():
    """Test DocumentsConfig validation"""
    print("\nTesting DocumentsConfig validation...")

    # Valid config
    config = DocumentsConfig(chunk_size=1000, chunk_overlap=200)
    print("  Valid config: PASS")

    # Invalid: overlap >= chunk_size
    try:
        DocumentsConfig(chunk_size=100, chunk_overlap=100)
        assert False, "Should have raised ConfigValidationError for overlap >= chunk_size"
    except ConfigValidationError:
        pass


def test_strategies_config_validation():
    """Test StrategiesConfig validation"""
    print("\nTesting StrategiesConfig validation...")

    # Valid config
    config = StrategiesConfig(simple_max=3, medium_max=7, complex_min=8)
    print("  Valid config: PASS")

    # Invalid: simple_max >= medium_max
    try:
        StrategiesConfig(simple_max=7, medium_max=7, complex_min=8)
        assert False, "Should have raised ConfigValidationError for simple_max >= medium_max"
    except ConfigValidationError:
        pass


def test_self_rag_config_validation():
    """Test SelfRAGConfig validation"""
    print("\nTesting SelfRAGConfig validation...")

    # Valid config
    config = SelfRAGConfig(max_regenerations=2, quality_threshold=0.5)
    print("  Valid config: PASS")

    # Invalid quality_threshold > 1
    try:
        SelfRAGConfig(quality_threshold=1.5)
        assert False, "Should have raised ConfigValidationError for quality_threshold=1.5"
    except ConfigValidationError:
        pass


def test_cache_config_validation():
    """Test CacheConfig validation"""
    print("\nTesting CacheConfig validation...")

    # Valid config
    config = CacheConfig(ttl_seconds=3600, query_cache_capacity=100)
    print("  Valid config: PASS")

    # Invalid ttl_seconds
    try:
        CacheConfig(ttl_seconds=-1)
        assert False, "Should have raised ConfigValidationError for ttl_seconds=-1"
    except ConfigValidationError:
        pass


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
