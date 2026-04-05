"""
Caching System for RAG
Provides LRU cache with semantic similarity matching for query response caching.
Similar queries (e.g., "What is ML?" and "What is machine learning?") hit the cache.
"""

import hashlib
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
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
    semantic_hits: int = 0  # Hits from semantic similarity (not exact match)

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
            "semantic_hits": self.semantic_hits,
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
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None if not found or expired."""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired():
                del self._cache[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1

            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache, evicting LRU entry if at capacity."""
        with self._lock:
            now = time.time()
            effective_ttl = ttl if ttl is not None else self.default_ttl

            if key in self._cache:
                self._cache.move_to_end(key)
                entry = self._cache[key]
                entry.value = value
                entry.last_accessed = now
                entry.access_count += 1
            else:
                while len(self._cache) >= self.capacity:
                    self._cache.popitem(last=False)
                    self.stats.evictions += 1

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
    Cache for query -> response mappings with semantic similarity matching.

    Uses embedding similarity to match queries that are semantically equivalent
    but worded differently. Falls back to exact-match hashing when no embedding
    function is provided.

    Example: "What is ML?" will hit a cache entry for "What is machine learning?"
    if their embedding cosine similarity exceeds the threshold (default 0.92).
    """

    def __init__(
        self,
        capacity: int = 500,
        ttl: float = 3600,
        similarity_threshold: float = 0.92,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Args:
            capacity: Maximum cached queries
            ttl: Default TTL in seconds
            similarity_threshold: Minimum cosine similarity for a semantic cache hit (0.0-1.0)
            embed_fn: Function that takes a string and returns an embedding vector.
                      If None, falls back to exact-match hashing only.
        """
        self._cache = LRUCache(capacity=capacity, default_ttl=ttl)
        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self._lock = threading.RLock()
        # Store query embeddings keyed by cache key for similarity search
        self._embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        # Map from cache key -> original query for debugging
        self._queries: OrderedDict[str, str] = OrderedDict()
        self.stats = CacheStats()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing"""
        return " ".join(query.lower().strip().split())

    def _query_hash(self, query: str) -> str:
        """Generate hash for a query"""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_semantic_match(self, query_embedding: np.ndarray) -> Optional[str]:
        """Find the best semantic match among cached embeddings.

        Returns the cache key of the best match if similarity exceeds threshold,
        or None if no match found.
        """
        best_key = None
        best_sim = 0.0

        for key, cached_emb in self._embeddings.items():
            sim = self._cosine_similarity(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_sim >= self.similarity_threshold:
            return best_key
        return None

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query.

        First tries exact-match hash lookup, then falls back to semantic
        similarity search if an embedding function is available.
        """
        with self._lock:
            # Fast path: exact match
            key = self._query_hash(query)
            result = self._cache.get(key)
            if result is not None:
                self.stats.hits += 1
                return result

            # Slow path: semantic similarity
            if self._embed_fn and self._embeddings:
                try:
                    query_embedding = np.array(self._embed_fn(query))
                    match_key = self._find_semantic_match(query_embedding)
                    if match_key is not None:
                        result = self._cache.get(match_key)
                        if result is not None:
                            self.stats.hits += 1
                            self.stats.semantic_hits += 1
                            if logger.isEnabledFor(logging.DEBUG):
                                original_query = self._queries.get(match_key, "?")
                                logger.debug(
                                    f"[SEMANTIC CACHE HIT] '{query[:50]}' matched '{original_query[:50]}'"
                                )
                            return result
                except Exception as e:
                    logger.warning(f"Semantic cache lookup failed: {e}")

            self.stats.misses += 1
            return None

    def put(
        self,
        query: str,
        response: str,
        documents: List[str] = None,
        strategy: str = None,
        metadata: Dict = None
    ) -> None:
        """Cache a query response with its embedding for future semantic matching."""
        with self._lock:
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
            self._queries[key] = query

            # Store embedding for semantic matching
            if self._embed_fn:
                try:
                    embedding = np.array(self._embed_fn(query))
                    self._embeddings[key] = embedding
                    # Evict old embeddings to match cache capacity
                    while len(self._embeddings) > self._cache.capacity:
                        self._embeddings.popitem(last=False)
                        self._queries.popitem(last=False)
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for cache: {e}")

    def invalidate(self, query: str) -> bool:
        """Invalidate a specific query cache"""
        with self._lock:
            key = self._query_hash(query)
            self._embeddings.pop(key, None)
            self._queries.pop(key, None)
            return self._cache.delete(key)

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        return self._cache.cleanup_expired()

    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            self._queries.clear()


class RAGCacheManager:
    """
    Unified cache manager for RAG systems.

    Provides semantic query caching — similar queries return cached responses
    without re-running retrieval and generation. Uses embedding similarity
    so "What is ML?" matches "What is machine learning?".

    Features:
    - Semantic similarity matching via embeddings
    - Exact-match fast path for identical queries
    - Background cleanup of expired entries
    - Statistics tracking
    """

    def __init__(
        self,
        query_cache_capacity: int = 500,
        query_cache_ttl: float = 3600,
        similarity_threshold: float = 0.92,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        enable_auto_cleanup: bool = True,
        cleanup_interval: float = 300,
    ):
        """
        Args:
            query_cache_capacity: Maximum cached query responses
            query_cache_ttl: TTL for cached responses in seconds
            similarity_threshold: Cosine similarity threshold for semantic matching
            embed_fn: Embedding function for semantic cache. If None, uses exact-match only.
            enable_auto_cleanup: Enable background cleanup of expired entries
            cleanup_interval: Seconds between cleanup runs
        """
        self.query_cache = SemanticQueryCache(
            capacity=query_cache_capacity,
            ttl=query_cache_ttl,
            similarity_threshold=similarity_threshold,
            embed_fn=embed_fn,
        )
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
        """Cleanup expired entries"""
        counts = {
            "query_cache": self.query_cache.cleanup_expired(),
        }
        total = sum(counts.values())
        if total > 0:
            logger.debug(f"Cleaned up {total} expired cache entries: {counts}")
        return counts

    def get_query_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached query response (exact match or semantic similarity)"""
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

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "query_cache": self.query_cache.stats.to_dict(),
            "summary": {
                "total_hits": self.query_cache.stats.hits,
                "total_misses": self.query_cache.stats.misses,
                "semantic_hits": self.query_cache.stats.semantic_hits,
            }
        }

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        stats = self.query_cache.stats
        total = stats.hits + stats.misses
        return stats.hits / total if total > 0 else 0.0

    def clear_all(self) -> None:
        """Clear all caches"""
        self.query_cache.clear()

    def shutdown(self) -> None:
        """Shutdown the cache manager, waiting for cleanup to finish"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=10.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Cache cleanup thread did not stop within timeout")
