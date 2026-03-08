"""
Tests for the RAG Caching System: LRU cache, TTL expiration, semantic cache, and cache stats.
All tests are fast and deterministic - no external dependencies required.
"""

import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.core.caching_system import (
    CacheEntry,
    CacheStats,
    LRUCache,
    SemanticQueryCache,
    VectorSearchCache,
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
        stats = CacheStats(hits=5, misses=5, evictions=2, expirations=1)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["expirations"] == 1
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

    def test_generate_key_string(self):
        cache = LRUCache()
        key = cache._generate_key("test string")
        assert isinstance(key, str)
        assert len(key) == 32

    def test_generate_key_non_string(self):
        cache = LRUCache()
        key = cache._generate_key({"key": "value"})
        assert isinstance(key, str)
        assert len(key) == 32


# --- SemanticQueryCache tests ---

class TestSemanticQueryCache:
    def test_put_and_get(self):
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

    def test_miss_for_different_query(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "answer")
        result = cache.get("What is deep learning?")
        assert result is None

    def test_invalidate(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("What is ML?", "answer")
        assert cache.invalidate("What is ML?") is True
        assert cache.get("What is ML?") is None

    def test_stats_property(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("q1", "a1")
        cache.get("q1")
        cache.get("q2")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_clear(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("q1", "a1")
        cache.clear()
        assert cache.get("q1") is None

    def test_stores_documents_and_metadata(self):
        cache = SemanticQueryCache(capacity=10)
        cache.put("q1", "a1", documents=["doc1"], strategy="hyde", metadata={"score": 0.9})
        result = cache.get("q1")
        assert result["documents"] == ["doc1"]
        assert result["metadata"] == {"score": 0.9}


# --- VectorSearchCache tests ---

class TestVectorSearchCache:
    def test_put_and_get_search_results(self):
        cache = VectorSearchCache(capacity=10)
        results = [("doc1", 0.9), ("doc2", 0.8)]
        cache.put_search_results("test query", 5, results)
        cached = cache.get_search_results("test query", 5)
        assert cached == results

    def test_different_k_is_different_key(self):
        cache = VectorSearchCache(capacity=10)
        cache.put_search_results("test", 5, [("doc1", 0.9)])
        assert cache.get_search_results("test", 10) is None

    def test_filter_criteria_affects_key(self):
        cache = VectorSearchCache(capacity=10)
        cache.put_search_results("test", 5, [("doc1", 0.9)], filter_criteria={"source": "A"})
        assert cache.get_search_results("test", 5) is None
        assert cache.get_search_results("test", 5, filter_criteria={"source": "A"}) is not None

    def test_embedding_cache(self):
        cache = VectorSearchCache(capacity=10)
        embedding = [0.1, 0.2, 0.3]
        cache.put_embedding("test text", embedding)
        cached = cache.get_embedding("test text")
        assert cached == embedding

    def test_embedding_cache_miss(self):
        cache = VectorSearchCache(capacity=10)
        assert cache.get_embedding("missing") is None

    def test_stats(self):
        cache = VectorSearchCache(capacity=10)
        stats = cache.stats
        assert "search_cache" in stats
        assert "embedding_cache" in stats

    def test_clear(self):
        cache = VectorSearchCache(capacity=10)
        cache.put_search_results("q", 5, [("d", 0.9)])
        cache.put_embedding("text", [0.1])
        cache.clear()
        assert cache.get_search_results("q", 5) is None
        assert cache.get_embedding("text") is None


# --- RAGCacheManager tests ---

class TestRAGCacheManager:
    def test_query_response_round_trip(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("What is Python?", "A language", strategy="baseline")
        result = manager.get_query_response("What is Python?")
        assert result is not None
        assert result["response"] == "A language"

    def test_search_results_round_trip(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        results = [("doc1", 0.95)]
        manager.cache_search_results("query", 5, results)
        cached = manager.get_search_results("query", 5)
        assert cached == results

    def test_embedding_round_trip(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        emb = [0.1, 0.2, 0.3, 0.4]
        manager.cache_embedding("text", emb)
        cached = manager.get_embedding("text")
        assert cached == emb

    def test_get_stats(self):
        manager = RAGCacheManager(enable_auto_cleanup=False)
        manager.cache_query_response("q1", "a1")
        manager.get_query_response("q1")
        stats = manager.get_stats()
        assert stats["query_cache"]["hits"] == 1
        assert "summary" in stats

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
        manager.cache_search_results("s1", 5, [("d", 0.9)])
        manager.cache_embedding("e1", [0.1])
        manager.clear_all()
        assert manager.get_query_response("q1") is None
        assert manager.get_search_results("s1", 5) is None
        assert manager.get_embedding("e1") is None

    def test_shutdown(self):
        manager = RAGCacheManager(enable_auto_cleanup=True, cleanup_interval=100)
        manager.shutdown()
        # Should not raise

    def test_cleanup_expired(self):
        manager = RAGCacheManager(
            enable_auto_cleanup=False,
            query_cache_ttl=0.01,
        )
        manager.cache_query_response("q1", "a1")
        time.sleep(0.02)
        counts = manager._cleanup_expired()
        assert counts["query_cache"] >= 1
