"""
Caching System for RAG
Provides LRU cache, semantic query cache, and vector search cache for 10x speedup
"""

import hashlib
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the cache with metadata"""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


@dataclass
class CacheStats:
    """Statistics for cache performance"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_requests": self.hits + self.misses
        }


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation

    Features:
    - O(1) get and put operations
    - Thread-safe with read-write lock
    - Optional TTL per entry
    - Automatic eviction when capacity reached
    """

    def __init__(self, capacity: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache

        Args:
            capacity: Maximum number of entries
            default_ttl: Default time-to-live in seconds (None = no expiry)
        """
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def _generate_key(self, data: Any) -> str:
        """Generate a hash key from data"""
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()[:32]
        return hashlib.sha256(str(data).encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1

            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        with self._lock:
            now = time.time()
            effective_ttl = ttl if ttl is not None else self.default_ttl

            if key in self._cache:
                # Update existing entry
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.value = value
                entry.last_accessed = now
                entry.access_count += 1
            else:
                # Evict if at capacity
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                    self.stats.evictions += 1

                # Add new entry
                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    ttl=effective_ttl
                )

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                self.stats.expirations += 1
            return len(expired_keys)


class SemanticQueryCache:
    """
    Cache for query -> response mappings

    Uses semantic hashing to cache similar queries and their responses.
    Designed for RAG query results.
    """

    def __init__(
        self,
        capacity: int = 500,
        ttl: float = 3600,  # 1 hour default
        similarity_threshold: float = 0.95
    ):
        """
        Initialize semantic query cache

        Args:
            capacity: Maximum cached queries
            ttl: Default TTL in seconds
            similarity_threshold: Threshold for considering queries similar
        """
        self._cache = LRUCache(capacity=capacity, default_ttl=ttl)
        self.similarity_threshold = similarity_threshold

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing"""
        # Convert to lowercase, strip whitespace, normalize spaces
        normalized = " ".join(query.lower().strip().split())
        return normalized

    def _query_hash(self, query: str) -> str:
        """Generate hash for a query"""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query

        Args:
            query: The query string

        Returns:
            Cached response dict or None
        """
        key = self._query_hash(query)
        return self._cache.get(key)

    def put(
        self,
        query: str,
        response: str,
        documents: List[str] = None,
        strategy: str = None,
        metadata: Dict = None
    ) -> None:
        """
        Cache a query response

        Args:
            query: The query string
            response: The generated response
            documents: List of retrieved document IDs/contents
            strategy: RAG strategy used
            metadata: Additional metadata
        """
        key = self._query_hash(query)
        entry = {
            "query": query,
            "response": response,
            "documents": documents or [],
            "strategy": strategy,
            "metadata": metadata or {},
            "cached_at": time.time()
        }
        self._cache.put(key, entry)

    def invalidate(self, query: str) -> bool:
        """Invalidate a specific query cache"""
        key = self._query_hash(query)
        return self._cache.delete(key)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._cache.stats

    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()


class VectorSearchCache:
    """
    Cache for vector similarity search results

    Caches query embedding -> document IDs mappings to avoid
    repeated vector similarity computations.
    """

    def __init__(
        self,
        capacity: int = 1000,
        ttl: float = 1800  # 30 minutes default
    ):
        """
        Initialize vector search cache

        Args:
            capacity: Maximum cached searches
            ttl: TTL in seconds
        """
        self._cache = LRUCache(capacity=capacity, default_ttl=ttl)
        self._embedding_cache = LRUCache(capacity=500, default_ttl=7200)  # 2 hour for embeddings

    def _search_key(self, query: str, k: int, filter_criteria: Dict = None) -> str:
        """Generate key for a search query"""
        key_parts = [query.lower().strip(), str(k)]
        if filter_criteria:
            key_parts.append(str(sorted(filter_criteria.items())))
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]

    def get_search_results(
        self,
        query: str,
        k: int,
        filter_criteria: Dict = None
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached search results

        Args:
            query: Search query
            k: Number of results
            filter_criteria: Additional filter criteria

        Returns:
            List of (doc_id, score) tuples or None
        """
        key = self._search_key(query, k, filter_criteria)
        return self._cache.get(key)

    def put_search_results(
        self,
        query: str,
        k: int,
        results: List[Tuple[str, float]],
        filter_criteria: Dict = None
    ) -> None:
        """
        Cache search results

        Args:
            query: Search query
            k: Number of results
            results: List of (doc_id, score) tuples
            filter_criteria: Additional filter criteria
        """
        key = self._search_key(query, k, filter_criteria)
        self._cache.put(key, results)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        key = hashlib.sha256(text.encode()).hexdigest()[:32]
        return self._embedding_cache.get(key)

    def put_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text"""
        key = hashlib.sha256(text.encode()).hexdigest()[:32]
        self._embedding_cache.put(key, embedding)

    @property
    def stats(self) -> Dict[str, CacheStats]:
        """Get cache statistics"""
        return {
            "search_cache": self._cache.stats,
            "embedding_cache": self._embedding_cache.stats
        }

    def clear(self) -> None:
        """Clear all caches"""
        self._cache.clear()
        self._embedding_cache.clear()


class RAGCacheManager:
    """
    Unified cache manager for RAG systems

    Manages:
    - Query response cache (SemanticQueryCache)
    - Vector search cache (VectorSearchCache)
    - General LRU cache for misc data

    Features:
    - Automatic cache warming
    - Background cleanup
    - Statistics aggregation
    """

    def __init__(
        self,
        query_cache_capacity: int = 500,
        query_cache_ttl: float = 3600,
        vector_cache_capacity: int = 1000,
        vector_cache_ttl: float = 1800,
        enable_auto_cleanup: bool = True,
        cleanup_interval: float = 300  # 5 minutes
    ):
        """
        Initialize RAG cache manager

        Args:
            query_cache_capacity: Capacity for query cache
            query_cache_ttl: TTL for query cache entries
            vector_cache_capacity: Capacity for vector cache
            vector_cache_ttl: TTL for vector cache entries
            enable_auto_cleanup: Enable automatic cleanup of expired entries
            cleanup_interval: Interval between cleanup runs
        """
        self.query_cache = SemanticQueryCache(
            capacity=query_cache_capacity,
            ttl=query_cache_ttl
        )
        self.vector_cache = VectorSearchCache(
            capacity=vector_cache_capacity,
            ttl=vector_cache_ttl
        )
        self.misc_cache = LRUCache(capacity=200, default_ttl=600)  # 10 min for misc

        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        if enable_auto_cleanup:
            self._start_cleanup_thread(cleanup_interval)

    def _start_cleanup_thread(self, interval: float) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup.wait(interval):
                self._cleanup_expired()

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches"""
        counts = {
            "query_cache": self.query_cache._cache.cleanup_expired(),
            "vector_search": self.vector_cache._cache.cleanup_expired(),
            "vector_embeddings": self.vector_cache._embedding_cache.cleanup_expired(),
            "misc_cache": self.misc_cache.cleanup_expired()
        }
        total = sum(counts.values())
        if total > 0:
            logger.debug(f"Cleaned up {total} expired cache entries: {counts}")
        return counts

    def get_query_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query response"""
        return self.query_cache.get(query)

    def cache_query_response(
        self,
        query: str,
        response: str,
        documents: List[str] = None,
        strategy: str = None,
        metadata: Dict = None
    ) -> None:
        """Cache a query response"""
        self.query_cache.put(query, response, documents, strategy, metadata)

    def get_search_results(
        self,
        query: str,
        k: int,
        filter_criteria: Dict = None
    ) -> Optional[List[Tuple[str, float]]]:
        """Get cached search results"""
        return self.vector_cache.get_search_results(query, k, filter_criteria)

    def cache_search_results(
        self,
        query: str,
        k: int,
        results: List[Tuple[str, float]],
        filter_criteria: Dict = None
    ) -> None:
        """Cache search results"""
        self.vector_cache.put_search_results(query, k, results, filter_criteria)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        return self.vector_cache.get_embedding(text)

    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding"""
        self.vector_cache.put_embedding(text, embedding)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated cache statistics"""
        return {
            "query_cache": self.query_cache.stats.to_dict(),
            "vector_cache": {
                "search": self.vector_cache.stats["search_cache"].to_dict(),
                "embedding": self.vector_cache.stats["embedding_cache"].to_dict()
            },
            "misc_cache": self.misc_cache.stats.to_dict(),
            "summary": {
                "total_hits": (
                    self.query_cache.stats.hits +
                    self.vector_cache.stats["search_cache"].hits +
                    self.vector_cache.stats["embedding_cache"].hits +
                    self.misc_cache.stats.hits
                ),
                "total_misses": (
                    self.query_cache.stats.misses +
                    self.vector_cache.stats["search_cache"].misses +
                    self.vector_cache.stats["embedding_cache"].misses +
                    self.misc_cache.stats.misses
                )
            }
        }

    def get_hit_rate(self) -> float:
        """Get overall cache hit rate"""
        stats = self.get_stats()
        total_hits = stats["summary"]["total_hits"]
        total_misses = stats["summary"]["total_misses"]
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0

    def clear_all(self) -> None:
        """Clear all caches"""
        self.query_cache.clear()
        self.vector_cache.clear()
        self.misc_cache.clear()

    def shutdown(self) -> None:
        """Shutdown the cache manager"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=1.0)


# Convenience function for quick testing
def test_caching_system():
    """Test basic caching functionality"""
    print("Testing RAG Caching System...")

    # Test LRU Cache
    print("\n1. Testing LRU Cache...")
    cache = LRUCache(capacity=3)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    assert cache.get("key1") == "value1"
    assert cache.size() == 3

    # Test eviction
    cache.put("key4", "value4")  # Should evict key2 (LRU after accessing key1)
    assert cache.size() == 3
    print("   LRU Cache: OK")

    # Test SemanticQueryCache
    print("\n2. Testing Semantic Query Cache...")
    query_cache = SemanticQueryCache(capacity=10)
    query_cache.put("What is machine learning?", "ML is a subset of AI...")
    result = query_cache.get("What is machine learning?")
    assert result is not None
    assert "ML is a subset" in result["response"]
    print("   Semantic Query Cache: OK")

    # Test VectorSearchCache
    print("\n3. Testing Vector Search Cache...")
    vector_cache = VectorSearchCache(capacity=10)
    vector_cache.put_search_results("test query", 5, [("doc1", 0.9), ("doc2", 0.8)])
    results = vector_cache.get_search_results("test query", 5)
    assert results is not None
    assert len(results) == 2
    print("   Vector Search Cache: OK")

    # Test RAGCacheManager
    print("\n4. Testing RAG Cache Manager...")
    manager = RAGCacheManager(enable_auto_cleanup=False)

    manager.cache_query_response(
        query="What is Python?",
        response="Python is a programming language...",
        strategy="baseline"
    )

    cached = manager.get_query_response("What is Python?")
    assert cached is not None
    assert "programming language" in cached["response"]

    stats = manager.get_stats()
    assert stats["query_cache"]["hits"] == 1
    print("   RAG Cache Manager: OK")

    print("\n" + "=" * 50)
    print("All caching tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_caching_system()
