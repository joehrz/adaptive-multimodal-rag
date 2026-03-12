import re
import logging
from typing import List, Optional

import numpy as np
from langchain.schema import Document

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Splits documents into chunks based on semantic similarity between sentences.
    Places chunk boundaries where embedding similarity between consecutive
    sentences drops below a percentile-based threshold.
    """

    def __init__(
        self,
        embeddings,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        similarity_threshold_percentile: int = 25,
    ):
        self.embeddings = embeddings
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold_percentile = similarity_threshold_percentile

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        if not similarities:
            return []
        threshold = float(np.percentile(similarities, self.similarity_threshold_percentile))
        return [i for i, sim in enumerate(similarities) if sim < threshold]

    def _chunk_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        sentence_embeddings = self.embeddings.embed_documents(sentences)
        sentence_embeddings = [np.array(e) for e in sentence_embeddings]

        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = self._cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
            similarities.append(sim)

        breakpoints = self._find_breakpoints(similarities)

        # Build chunks from sentences, splitting at breakpoints
        chunks = []
        current_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            current_text = " ".join(current_sentences)

            # Check if adding this sentence would exceed max_chunk_size
            next_text = current_text + " " + sentences[i]
            if len(next_text) > self.max_chunk_size and len(current_text) >= self.min_chunk_size:
                chunks.append(current_text)
                current_sentences = [sentences[i]]
                continue

            # Split at breakpoint if current chunk is large enough
            if (i - 1) in breakpoints and len(current_text) >= self.min_chunk_size:
                chunks.append(current_text)
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])

        # Add remaining sentences
        if current_sentences:
            remaining = " ".join(current_sentences)
            # Merge small trailing chunk with previous if possible
            if len(remaining) < self.min_chunk_size and chunks:
                merged = chunks[-1] + " " + remaining
                if len(merged) <= self.max_chunk_size:
                    chunks[-1] = merged
                else:
                    chunks.append(remaining)
            else:
                chunks.append(remaining)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result = []
        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue

            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = i
                result.append(Document(page_content=chunk, metadata=metadata))

        logger.info(f"Semantic chunking: {len(documents)} docs -> {len(result)} chunks")
        return result
