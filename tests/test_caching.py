"""
Tests for the RAG Caching System: LRU cache, semantic query cache, and cache stats.
All tests are fast and deterministic - no external dependencies required.
"""

import sys
import time
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.core.caching_system import (
    CacheEntry,
    CacheStats,
    LRUCache,
    SemanticQueryCache,
    RAGCacheManager,
)


# --- CacheEntry tests ---

class TestCacheEntry:
    def test_not_expired_without_ttl(self):
        entry = CacheEntry(value="test", created_at=time.time(), last_accessed=time.time())
        assert entry.is_expired() is False

    def test_not_expired_within_ttl(self):
        entry = CacheEntry(
            value="test", created_at=time.time(), last_accessed=time.time(), ttl=3600
        )
        assert entry.is_expired() is False

    def test_expired_after_ttl(self):
        entry = CacheEntry(
            value="test", created_at=time.time() - 10, last_accessed=time.time(), ttl=5
        )
        assert entry.is_expired() is True

    def test_default_access_count(self):
        entry = CacheEntry(value="x", created_at=0, last_accessed=0)
        assert entry.access_count == 0


# --- CacheStats tests ---

class TestCacheStats:
    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == pytest.approx(0.3)

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_to_dict(self):
        stats = CacheStats(hits=5, misses=5, evictions=2, expirations=1, semantic_hits=2)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["expirations"] == 1
        assert d["semantic_hits"] == 2
        assert d["hit_rate"] == "50.00%"
        assert d["total_requests"] == 10


# --- LRUCache tests ---

class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(capacity=10)
        cache.put("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing_key(self):
        cache = LRUCache(capacity=10)
        assert cache.get("nonexistent") is None
        assert cache.stats.misses == 1

    def test_eviction_at_capacity(self):
        cache = LRUCache(capacity=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")  # should evict k1 (LRU)
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"
        assert cache.stats.evictions == 1

    def test_lru_order_updated_on_access(self):
        cache = LRUCache(capacity=2)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.get("k1")  # access k1, making k2 the LRU
        cache.put("k3", "v3")  # should evict k2
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"

    def test_update_existing_key(self):
        cache = LRUCache(capacity=10)
        cache.put("k1", "v1")
        cache.put("k1", "v2")
        assert cache.get("k1") == "v2"
        assert cache.size() == 1

    def test_ttl_expiration(self):
        cache = LRUCache(capacity=10, default_ttl=0.01)
        cache.put("k1", "v1")
        time.sleep(0.02)
        assert cache.get("k1") is None
        assert cache.stats.expirations == 1

    def test_per_entry_ttl_overrides_default(self):
        cache = LRUCache(capacity=10, default_ttl=3600)
        cache.put("k1", "v1", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k1") is None

    def test_delete(self):
        cache = LRUCache(capacity=10)
        cache.put("k1", "v1")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None
        assert cache.delete("k1") is False

    def test_clear(self):
        cache = LRUCache(capacity=10)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.clear()
        assert cache.size() == 0

    def test_size(self):
        cache = LRUCache(capacity=10)
        assert cache.size() == 0
        cache.put("k1", "v1")
        assert cache.size() == 1
        cache.put("k2", "v2")
        assert cache.size() == 2

    def test_cleanup_expired(self):
        cache = LRUCache(capacity=10)
        cache.put("k1", "v1", ttl=0.01)
        cache.put("k2", "v2")  # no TTL
        time.sleep(0.02)
        expired_count = cache.cleanup_expired()
        assert expired_count == 1
        assert cache.size() == 1
        assert cache.get("k2") == "v2"

    def test_stats_tracking(self):
        cache = LRUCache(capacity=2)
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1


# --- SemanticQueryCache tests ---

def _mock_embed_fn(text):
    """Simple deterministic embedding for testing.
    Similar texts get similar vectors."""
    np.random.seed(hash(text.lower().strip()) % 2**31)
    return np.random.randn(64).tolist()


def _similar_embed_fn(text):
    """Embedding function where specific queries map to similar vectors."""
    base = np.zeros(64)
    text_lower = text.lower().strip()

    if "machine learning" in text_lower or "ml" in text_lower:
        base[0] = 1.0
        base[1] = 0.9
        # Add small noise based on exact text so they're similar but not identical
        np.random.seed(hash(text_lower) % 2**31)
        base += np.random.randn(64) * 0.05
    elif "deep learning" in text_lower:
        base[0] = 0.8
        base[2] = 1.0
        np.random.seed(hash(text_lower) % 2**31)
        base += np.random.randn(64) * 0.05
    else:
        np.random.seed(hash(text_lower) % 2**31)
        base = np.random.randn(64)

    return base.tolist()


class TestSemanticQueryCache:
    def test_put_and_get_exact_match(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "Machine learning is...", strategy="baseline")
        result = cache.get("What is ML?")
        assert result is not None
        assert result["response"] == "Machine learning is..."
        assert result["strategy"] == "baseline"

    def test_normalization_case_insensitive(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "answer")
        result = cache.get("WHAT IS ML?")
        assert result is not None

    def test_normalization_strips_whitespace(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("  What  is  ML?  ", "answer")
        result = cache.get("What is ML?")
        assert result is not None

    def test_miss_for_different_query_without_embeddings(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "answer")
        result = cache.get("What is deep learning?")
        assert result is None

    def test_semantic_match_similar_queries(self):
        cache = SemanticQueryCache(
            capacity=10,
            similarity_threshold=0.85,
            embed_fn=_similar_embed_fn,
        )
        cache.put("What is machine learning?", "ML is a subset of AI")
        # "What is ML?" should semantically match
        result = cache.get("What is ML?")
        assert result is not None
        assert result["response"] == "ML is a subset of AI"
        assert cache.stats.semantic_hits == 1

    def test_semantic_miss_for_dissimilar_queries(self):
        cache = SemanticQueryCache(
            capacity=10,
            similarity_threshold=0.85,
            embed_fn=_similar_embed_fn,
        )
        cache.put("What is machine learning?", "ML is a subset of AI")
        # "What is deep learning?" should NOT match (different topic cluster)
        result = cache.get("What is deep learning?")
        assert result is None

    def test_invalidate(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "answer")
        assert cache.invalidate("What is ML?") is True
        assert cache.get("What is ML?") is None

    def test_stats_tracking(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("q1", "a1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_clear(self):
        cache = SemanticQueryCache(capacity=10, embed_fn=_mock_embed_fn)
        cache.put("q1", "a1")
        cache.clear()
        assert cache.get("q1") is None

    def test_stores_documents_and_metadata(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("q1", "a1", documents=["doc1"], strategy="hyde", metadata={"score": 0.9})
        result = cache.get("q1")
        assert result["documents"] == ["doc1"]
        assert result["metadata"] == {"score": 0.9}

    def test_embed_fn_failure_falls_back_to_exact_match(self):
        def bad_embed_fn(text):
            raise RuntimeError("embedding failed")

        cache = SemanticQueryCache(capacity=10, embed_fn=bad_embed_fn)
        cache.put("What is ML?", "answer")
        # Exact match should still work
        result = cache.get("What is ML?")
        assert result is not None
        # Different query should miss (no semantic fallback)
        result = cache.get("What is machine learning?")
        assert result is None


# --- RAGCacheManager tests ---

class TestRAGCacheManager:
    def test_query_response_round_trip(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("What is Python?", "A language", strategy="baseline")
        result = manager.get_query_response("What is Python?")
        assert result is not None
        assert result["response"] == "A language"

    def test_semantic_match_with_embed_fn(self):
        manager = RAGCacheManager(
            enable_auto_cleanup=False,
            embed_fn=_similar_embed_fn,
            similarity_threshold=0.85,
        )
        manager.cache_query_response(
            "What is machine learning?", "ML is AI", strategy="baseline"
        )
        # Semantic match
        result = manager.get_query_response("What is ML?")
        assert result is not None
        assert result["response"] == "ML is AI"

    def test_get_stats(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("q1", "a1")
        manager.get_query_response("q1")
        stats = manager.get_stats()
        assert stats["query_cache"]["hits"] == 1
        assert "summary" in stats
        assert "semantic_hits" in stats["summary"]

    def test_get_hit_rate_empty(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        assert manager.get_hit_rate() == 0.0

    def test_get_hit_rate_nonzero(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("q1", "a1")
        manager.get_query_response("q1")  # hit
        manager.get_query_response("q2")  # miss
        rate = manager.get_hit_rate()
        assert rate > 0

    def test_clear_all(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("q1", "a1")
        manager.clear_all()
        assert manager.get_query_response("q1") is None

    def test_shutdown(self):
        manager = RAGCacheManager(enable_auto_cleanup=True, cleanup_interval=100)
        manager.shutdown()

    def test_cleanup_expired(self):
        manager = RAGCacheManager(
            enable_auto_cleanup=False,
            query_cache_ttl=0.01,
        )
        manager.cache_query_response("q1", "a1")
        time.sleep(0.02)
        counts = manager._cleanup_expired()
        assert counts["query_cache"] >= 1
