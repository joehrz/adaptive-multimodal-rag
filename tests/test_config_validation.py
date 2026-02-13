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
        print("  Invalid temperature (3.0): FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid temperature (3.0): PASS - {e}")

    # Invalid max_tokens
    try:
        LLMConfig(max_tokens=0)
        print("  Invalid max_tokens (0): FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid max_tokens (0): PASS - {e}")

    return True


def test_documents_config_validation():
    """Test DocumentsConfig validation"""
    print("\nTesting DocumentsConfig validation...")

    # Valid config
    config = DocumentsConfig(chunk_size=1000, chunk_overlap=200)
    print("  Valid config: PASS")

    # Invalid: overlap >= chunk_size
    try:
        DocumentsConfig(chunk_size=100, chunk_overlap=100)
        print("  Invalid overlap >= chunk_size: FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid overlap >= chunk_size: PASS - {e}")

    return True


def test_strategies_config_validation():
    """Test StrategiesConfig validation"""
    print("\nTesting StrategiesConfig validation...")

    # Valid config
    config = StrategiesConfig(simple_max=3, medium_max=7, complex_min=8)
    print("  Valid config: PASS")

    # Invalid: simple_max >= medium_max
    try:
        StrategiesConfig(simple_max=7, medium_max=7, complex_min=8)
        print("  Invalid simple_max >= medium_max: FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid simple_max >= medium_max: PASS - {e}")

    return True


def test_self_rag_config_validation():
    """Test SelfRAGConfig validation"""
    print("\nTesting SelfRAGConfig validation...")

    # Valid config
    config = SelfRAGConfig(max_regenerations=2, quality_threshold=0.5)
    print("  Valid config: PASS")

    # Invalid quality_threshold > 1
    try:
        SelfRAGConfig(quality_threshold=1.5)
        print("  Invalid quality_threshold (1.5): FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid quality_threshold (1.5): PASS - {e}")

    return True


def test_cache_config_validation():
    """Test CacheConfig validation"""
    print("\nTesting CacheConfig validation...")

    # Valid config
    config = CacheConfig(ttl_seconds=3600, query_cache_capacity=100)
    print("  Valid config: PASS")

    # Invalid ttl_seconds
    try:
        CacheConfig(ttl_seconds=-1)
        print("  Invalid ttl_seconds (-1): FAIL - should have raised error")
        return False
    except ConfigValidationError as e:
        print(f"  Invalid ttl_seconds (-1): PASS - {e}")

    return True


def run_all_tests():
    """Run all config validation tests"""
    print("=" * 70)
    print("CONFIG VALIDATION TESTS")
    print("=" * 70)

    results = []

    results.append(("LLMConfig", test_llm_config_validation()))
    results.append(("DocumentsConfig", test_documents_config_validation()))
    results.append(("StrategiesConfig", test_strategies_config_validation()))
    results.append(("SelfRAGConfig", test_self_rag_config_validation()))
    results.append(("CacheConfig", test_cache_config_validation()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
