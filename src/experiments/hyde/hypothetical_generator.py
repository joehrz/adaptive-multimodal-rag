"""
Hypothetical document generation for HyDE.

Handles prompt construction and LLM calls to generate hypothetical documents
that would answer a given query, as well as query-type-aware answer generation.
"""

import time
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Summarization detection keywords
_SUMMARIZATION_KEYWORDS = [
    'summarize', 'summary', 'summarise', 'overview', 'abstract',
    'main points', 'key points', 'key findings', 'key takeaways',
    'main contribution', 'main contributions', 'main idea',
    'tldr', 'recap', 'brief',
    'describe the paper', 'what does the paper say',
    'what is the paper about', 'what does this paper',
]


class HypotheticalGenerator:
    """Generates hypothetical documents and answers using a provided LLM callable."""

    def __init__(
        self,
        model: str,
        temperature: float,
        answer_temperature: float,
        max_tokens: int,
        hypothetical_max_tokens: int,
        llm_generate: Callable,
        verbose: bool = False,
    ):
        """
        Args:
            model: Model name for generation.
            temperature: Temperature for hypothetical document generation.
            answer_temperature: Temperature for final answer generation.
            max_tokens: Max tokens for final answer.
            hypothetical_max_tokens: Max tokens for hypothetical document.
            llm_generate: Callable that wraps ollama.generate (or equivalent).
            verbose: Enable verbose logging.
        """
        self.model = model
        self.temperature = temperature
        self.answer_temperature = answer_temperature
        self.max_tokens = max_tokens
        self.hypothetical_max_tokens = hypothetical_max_tokens
        self.verbose = verbose
        self._llm_generate = llm_generate

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        This is the core of HyDE - we create a fake but plausible answer
        to use for semantic retrieval.
        """
        prompt = f"""You are a knowledgeable assistant. Write a detailed, informative passage that would directly answer the following question.
Write as if this passage is from a textbook or authoritative source.
Do NOT say "I don't know" or ask questions - just write an informative passage.

Question: {query}

Informative passage:"""

        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self._llm_generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                        'num_predict': self.hypothetical_max_tokens,
                    }
                )

                return response['response'].strip()

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"HyDE generation failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 2s...")
                    time.sleep(2)
                else:
                    logger.error(f"Error generating hypothetical document after {max_retries} attempts: {last_error}")
                    raise RuntimeError(f"HyDE hypothetical generation failed after {max_retries} attempts: {last_error}") from last_error

    def detect_summarization_query(self, query: str) -> bool:
        """Detect if query is asking for a summary or overview"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in _SUMMARIZATION_KEYWORDS)

    def generate_answer(self, query: str, context: str, hypothetical: str = None) -> str:
        """Generate final answer using retrieved context with query-type-aware prompts"""

        if self.detect_summarization_query(query):
            prompt = f"""You are summarizing a document based on the provided context.
Synthesize the information from ALL provided documents into a summary.
Cover the main contributions, methodology, key findings, and conclusions.
Use the format [Document X] when referencing specific information.
Do NOT say you cannot summarize - work with the context provided.

Context from retrieved documents:
{context}

Request: {query}

Summary (with citations):"""
        else:
            prompt = f"""Answer the following question using the provided context.
Be accurate and cite information from the context when relevant.
If the context doesn't contain the answer, say so.

Context from retrieved documents:
{context}

Question: {query}

Answer:"""

        response = self._llm_generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.answer_temperature,
                'num_predict': self.max_tokens,
            }
        )

        return response['response'].strip()
