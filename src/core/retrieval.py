"""
Document retrieval, reranking, deduplication, and query detection.
Extracted from ollama_rag.py for modularity.
"""

import re
import hashlib
import logging
from typing import List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)


def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """Deduplicate chunks by content hash before adding to vector store."""
    if not chunks:
        return []

    seen_hashes = set()
    unique_chunks = []

    for chunk in chunks:
        content = chunk.page_content.strip()
        if not content:
            continue

        content_hash = hashlib.sha256(content.lower().encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)

    return unique_chunks


def deduplicate_documents(docs: List[Document], min_chars: int = 500) -> List[Document]:
    """Deduplicate documents by hashing the first min_chars of content."""
    if not docs:
        return []

    seen_content = set()
    unique_docs = []

    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue

        # Skip short chunks below threshold
        dedup_chars = min(min_chars, len(content))
        content_for_hash = content[:dedup_chars].lower()
        content_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()

        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


def detect_page_query(query: str) -> Optional[int]:
    """Detect if query is asking about a specific page number."""
    patterns = [
        r'page\s*(\d+)',
        r'p\.?\s*(\d+)',
        r'pg\.?\s*(\d+)',
    ]
    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))
    return None


def detect_metadata_query(query: str) -> bool:
    """Detect if query is asking about document metadata (title, authors, date)."""
    metadata_keywords = [
        'title of', 'paper title', 'what is the title',
        'who wrote', 'who are the authors', 'authors of',
        'publication date', 'year of publication',
        'which journal', 'which conference',
    ]
    metadata_patterns = [
        r'when was .* published',
        r'where was .* published',
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in metadata_keywords):
        return True
    return any(re.search(pat, query_lower) for pat in metadata_patterns)


def detect_summarization_query(query: str) -> bool:
    """Detect if query is asking for a summary or overview."""
    summarization_keywords = [
        'summarize', 'summary', 'summarise', 'overview', 'abstract',
        'main points', 'key points', 'key findings', 'key takeaways',
        'main contribution', 'main contributions', 'main idea',
        'tldr', 'recap', 'brief',
        'describe the paper', 'what does the paper say',
        'what is the paper about', 'what does this paper',
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in summarization_keywords)


def keyword_search(documents: List[Document], query: str, k: int = 5) -> List[Document]:
    """Supplement semantic search with keyword-based retrieval from stored documents."""
    if not documents:
        return []

    query_terms = [w.lower() for w in query.split() if len(w) > 3]
    if not query_terms:
        return []

    scored_chunks = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        if matches > 0:
            scored_chunks.append((matches, doc))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_chunks[:k]]


def retrieve_first_pages(
    documents: List[Document],
    vector_store,
    num_pages: int = 2,
    verbose: bool = False,
) -> List[Document]:
    """Retrieve chunks from the first pages of uploaded documents (for metadata queries)."""
    first_page_docs = []

    for doc in documents:
        page = doc.metadata.get('page')
        if page is not None and page <= num_pages:
            first_page_docs.append(doc)

    if first_page_docs:
        if verbose:
            logger.info(f"[METADATA] Found {len(first_page_docs)} chunks from first {num_pages} pages")
        return deduplicate_documents(first_page_docs)

    # Fallback: search vector store for abstract/introduction
    if vector_store:
        try:
            first_page_docs = vector_store.similarity_search(
                "abstract introduction title authors",
                k=5
            )
        except Exception:
            pass

    return deduplicate_documents(first_page_docs)


def rerank(query: str, documents: List[Document], reranker, top_k: int = 10,
           min_score_gap: float = 5.0, verbose: bool = False) -> List[Document]:
    """Rerank documents using cross-encoder model."""
    if not documents or not reranker:
        return documents[:top_k]

    pairs = [(query, doc.page_content[:512]) for doc in documents]

    try:
        scores = reranker.predict(pairs)

        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        # Filter out chunks that score far below the top result
        top_score_val = scored_docs[0][0] if scored_docs else 0
        score_floor = top_score_val - min_score_gap

        filtered = [(score, doc) for score, doc in scored_docs if score >= score_floor]

        # Always keep at least 1 document
        if not filtered and scored_docs:
            filtered = [scored_docs[0]]

        reranked = [doc for score, doc in filtered[:top_k]]

        if verbose:
            dropped = len(scored_docs) - len(filtered)
            bottom_score = filtered[-1][0] if filtered else 0
            logger.info(f"[RERANK] Reranked {len(documents)} -> {len(reranked)} docs "
                       f"(scores: {top_score_val:.3f} to {bottom_score:.3f}"
                       f"{f', dropped {dropped} below gap {min_score_gap}' if dropped > 0 else ''})")

        return reranked

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Falling back to original order.")
        return documents[:top_k]
