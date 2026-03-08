"""
Self-RAG Implementation with Reflection Tokens
Based on "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

Reflection tokens evaluate:
- Relevance: Are retrieved documents relevant to the query?
- Support: Is the answer supported by the documents?
- Utility: Is the answer useful for the user?
"""

import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Iterator, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from langchain.schema import Document

# Import config system
try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)


class RelevanceToken(Enum):
    """Relevance assessment of retrieved documents"""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"

    @property
    def score(self) -> float:
        """Numeric score for relevance"""
        scores = {
            RelevanceToken.RELEVANT: 1.0,
            RelevanceToken.PARTIALLY_RELEVANT: 0.5,
            RelevanceToken.IRRELEVANT: 0.0
        }
        return scores[self]

    @property
    def description(self) -> str:
        """Human-readable description"""
        descriptions = {
            RelevanceToken.RELEVANT: "Documents directly address the query",
            RelevanceToken.PARTIALLY_RELEVANT: "Documents contain some relevant information",
            RelevanceToken.IRRELEVANT: "Documents do not address the query"
        }
        return descriptions[self]


class SupportToken(Enum):
    """Support assessment - is the answer grounded in documents?"""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"

    @property
    def score(self) -> float:
        """Numeric score for support"""
        scores = {
            SupportToken.FULLY_SUPPORTED: 1.0,
            SupportToken.PARTIALLY_SUPPORTED: 0.5,
            SupportToken.NO_SUPPORT: 0.0
        }
        return scores[self]

    @property
    def description(self) -> str:
        """Human-readable description"""
        descriptions = {
            SupportToken.FULLY_SUPPORTED: "Answer is fully grounded in retrieved documents",
            SupportToken.PARTIALLY_SUPPORTED: "Answer is partially supported by documents",
            SupportToken.NO_SUPPORT: "Answer is not supported by retrieved documents"
        }
        return descriptions[self]


class UtilityToken(Enum):
    """Utility assessment - is the answer useful?"""
    USEFUL = "useful"
    SOMEWHAT_USEFUL = "somewhat_useful"
    NOT_USEFUL = "not_useful"

    @property
    def score(self) -> float:
        """Numeric score for utility"""
        scores = {
            UtilityToken.USEFUL: 1.0,
            UtilityToken.SOMEWHAT_USEFUL: 0.5,
            UtilityToken.NOT_USEFUL: 0.0
        }
        return scores[self]

    @property
    def description(self) -> str:
        """Human-readable description"""
        descriptions = {
            UtilityToken.USEFUL: "Answer fully addresses the user's question",
            UtilityToken.SOMEWHAT_USEFUL: "Answer partially addresses the question",
            UtilityToken.NOT_USEFUL: "Answer does not address the question"
        }
        return descriptions[self]


@dataclass
class ReflectionResult:
    """Complete reflection assessment"""
    relevance: RelevanceToken
    support: SupportToken
    utility: UtilityToken
    relevance_reasoning: str = ""
    support_reasoning: str = ""
    utility_reasoning: str = ""
    overall_score: float = 0.0
    needs_regeneration: bool = False
    reflection_time: float = 0.0

    def __post_init__(self):
        """Calculate overall score"""
        self.overall_score = (
            self.relevance.score * 0.3 +
            self.support.score * 0.4 +
            self.utility.score * 0.3
        )
        # Trigger regeneration if overall quality is low
        self.needs_regeneration = self.overall_score < 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "relevance": {
                "token": self.relevance.value,
                "score": self.relevance.score,
                "reasoning": self.relevance_reasoning
            },
            "support": {
                "token": self.support.value,
                "score": self.support.score,
                "reasoning": self.support_reasoning
            },
            "utility": {
                "token": self.utility.value,
                "score": self.utility.score,
                "reasoning": self.utility_reasoning
            },
            "overall_score": self.overall_score,
            "needs_regeneration": self.needs_regeneration,
            "reflection_time": self.reflection_time
        }

    def summary(self) -> str:
        """Get a brief summary of the reflection"""
        status = "PASS" if not self.needs_regeneration else "NEEDS IMPROVEMENT"
        return (
            f"[{status}] Score: {self.overall_score:.2f} | "
            f"Relevance: {self.relevance.value} | "
            f"Support: {self.support.value} | "
            f"Utility: {self.utility.value}"
        )


@dataclass
class SelfRAGResult:
    """Result from Self-RAG query with reflection"""
    query: str
    answer: str
    documents: List[Document]
    reflection: ReflectionResult
    regeneration_count: int = 0
    total_time: float = 0.0
    generation_time: float = 0.0
    was_regenerated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "document_count": len(self.documents),
            "reflection": self.reflection.to_dict(),
            "regeneration_count": self.regeneration_count,
            "total_time": self.total_time,
            "generation_time": self.generation_time,
            "was_regenerated": self.was_regenerated
        }


class OllamaSelfRAG:
    """
    Self-RAG implementation using Ollama

    Key features:
    - Reflection tokens for quality assessment
    - Automatic regeneration for low-quality answers
    - Streaming support for UI integration
    """

    def __init__(
        self,
        model: Optional[str] = None,
        reflection_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_regenerations: Optional[int] = None,
        regeneration_threshold: Optional[float] = None,
        verbose: Optional[bool] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        reflection_temperature: Optional[float] = None,
    ):
        """
        Initialize Self-RAG system

        Args:
            model: Main generation model
            reflection_model: Model for reflection (defaults to main model)
            temperature: Generation temperature
            max_tokens: Maximum tokens for generation
            max_regenerations: Maximum regeneration attempts
            regeneration_threshold: Score threshold for regeneration
            verbose: Enable verbose logging
            config: Optional Config object (uses global config if not provided)
            timeout: Timeout for LLM calls in seconds
            reflection_temperature: Temperature for reflection assessments
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.temperature = temperature if temperature is not None else config.llm.temperature
            self.max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
            self.max_regenerations = max_regenerations if max_regenerations is not None else config.strategies.self_rag.max_regenerations
            self.regeneration_threshold = regeneration_threshold if regeneration_threshold is not None else config.strategies.self_rag.quality_threshold
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.timeout = timeout if timeout is not None else config.llm.timeout
            self.reflection_temperature = reflection_temperature if reflection_temperature is not None else config.strategies.self_rag.reflection_temperature
        else:
            # Fallback to hardcoded defaults if no config available
            self.model = model or "qwen2.5:14b"
            self.temperature = temperature if temperature is not None else 0.3
            self.max_tokens = max_tokens or 1000
            self.max_regenerations = max_regenerations or 2
            self.regeneration_threshold = regeneration_threshold or 0.5
            self.verbose = verbose if verbose is not None else True
            self.timeout = timeout or 120
            self.reflection_temperature = reflection_temperature or 0.1

        self.reflection_model = reflection_model or self.model

        # Verify model availability
        try:
            available_models = ollama.list()
            model_names = [m.model for m in available_models.models]
            if self.model not in model_names:
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")
            if self.verbose:
                logger.info(f"Self-RAG initialized with model: {self.model}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def _generate_answer(self, query: str, documents: List[Document]) -> Tuple[str, float]:
        """Generate an answer from documents"""
        start_time = time.time()

        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents[:5])
        ])

        prompt = f"""Based on the following documents, answer the question accurately and completely.
If the documents don't contain relevant information, acknowledge this limitation.

Documents:
{context}

Question: {query}

Answer:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        )

        generation_time = time.time() - start_time
        return response['response'].strip(), generation_time

    def _parse_reflection_token(self, response_text: str, token_patterns: Dict[str, Any]) -> Tuple[Any, str]:
        """
        Robustly parse a reflection token from LLM response.

        Args:
            response_text: Raw LLM response text
            token_patterns: Dict mapping token enum values to their patterns

        Returns:
            Tuple of (matched_token, reasoning)
        """
        text_upper = response_text.upper()
        reasoning = ""

        # Try to extract reasoning first
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Try to match tokens using word boundaries and specific patterns
        # This prevents matching "NOT USEFUL" when looking for "USEFUL"
        for token, pattern_info in token_patterns.items():
            pattern = pattern_info.get('pattern')
            excludes = pattern_info.get('excludes', [])

            # Check exclusions first
            excluded = False
            for exclude in excludes:
                if re.search(exclude, text_upper):
                    excluded = True
                    break

            if excluded:
                continue

            # Check for the pattern
            if re.search(pattern, text_upper):
                return token, reasoning

        # Return default (first token, which should be the "negative" case)
        return list(token_patterns.keys())[-1], reasoning

    def _assess_relevance(self, query: str, documents: List[Document]) -> Tuple[RelevanceToken, str]:
        """Assess relevance of retrieved documents"""
        doc_summaries = "\n".join([
            f"- Doc {i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(documents[:3])
        ])

        prompt = f"""Assess whether the following documents are relevant to answering the query.

Query: {query}

Documents:
{doc_summaries}

Rate the relevance as one of:
- RELEVANT: Documents directly address the query
- PARTIALLY_RELEVANT: Documents contain some relevant information
- IRRELEVANT: Documents do not address the query

Provide your assessment in this format:
RELEVANCE: [your rating]
REASONING: [brief explanation]"""

        response = ollama.generate(
            model=self.reflection_model,
            prompt=prompt,
            options={
                'temperature': self.reflection_temperature,
                'num_predict': 150
            }
        )

        # Use robust token parsing
        token_patterns = {
            RelevanceToken.RELEVANT: {
                'pattern': r'\bRELEVANT\b',
                'excludes': [r'\bPARTIALLY[_\s]*RELEVANT\b', r'\bIRRELEVANT\b', r'\bNOT\s+RELEVANT\b']
            },
            RelevanceToken.PARTIALLY_RELEVANT: {
                'pattern': r'\bPARTIALLY[_\s]*RELEVANT\b',
                'excludes': []
            },
            RelevanceToken.IRRELEVANT: {
                'pattern': r'\b(IRRELEVANT|NOT\s+RELEVANT)\b',
                'excludes': []
            },
        }

        token, reasoning = self._parse_reflection_token(response['response'], token_patterns)
        return token, reasoning

    def _assess_support(self, answer: str, documents: List[Document]) -> Tuple[SupportToken, str]:
        """Assess whether answer is supported by documents"""
        doc_content = "\n".join([doc.page_content[:300] for doc in documents[:3]])

        prompt = f"""Assess whether the following answer is supported by the source documents.

Answer: {answer}

Source Documents:
{doc_content}

Rate the support as one of:
- FULLY_SUPPORTED: All claims in the answer are directly supported by the documents
- PARTIALLY_SUPPORTED: Some claims are supported, some are not
- NO_SUPPORT: The answer contains claims not found in the documents

Provide your assessment in this format:
SUPPORT: [your rating]
REASONING: [brief explanation]"""

        response = ollama.generate(
            model=self.reflection_model,
            prompt=prompt,
            options={
                'temperature': self.reflection_temperature,
                'num_predict': 150
            }
        )

        # Use robust token parsing
        token_patterns = {
            SupportToken.FULLY_SUPPORTED: {
                'pattern': r'\bFULLY[_\s]*SUPPORTED\b',
                'excludes': [r'\bPARTIALLY[_\s]*SUPPORTED\b', r'\bNO[_\s]*SUPPORT\b', r'\bNOT\s+SUPPORTED\b']
            },
            SupportToken.PARTIALLY_SUPPORTED: {
                'pattern': r'\bPARTIALLY[_\s]*SUPPORTED\b',
                'excludes': []
            },
            SupportToken.NO_SUPPORT: {
                'pattern': r'\b(NO[_\s]*SUPPORT|NOT\s+SUPPORTED|UNSUPPORTED)\b',
                'excludes': []
            },
        }

        token, reasoning = self._parse_reflection_token(response['response'], token_patterns)
        return token, reasoning

    def _assess_utility(self, query: str, answer: str) -> Tuple[UtilityToken, str]:
        """Assess whether answer is useful for the user"""
        prompt = f"""Assess whether the following answer is useful for answering the user's question.

Question: {query}

Answer: {answer}

Rate the utility as one of:
- USEFUL: Answer fully addresses the user's question with complete information
- SOMEWHAT_USEFUL: Answer partially addresses the question but may be incomplete
- NOT_USEFUL: Answer does not address the question or is too vague

Provide your assessment in this format:
UTILITY: [your rating]
REASONING: [brief explanation]"""

        response = ollama.generate(
            model=self.reflection_model,
            prompt=prompt,
            options={
                'temperature': self.reflection_temperature,
                'num_predict': 150
            }
        )

        # Use robust token parsing
        token_patterns = {
            UtilityToken.USEFUL: {
                'pattern': r'\bUSEFUL\b',
                'excludes': [r'\bSOMEWHAT[_\s]*USEFUL\b', r'\bNOT[_\s]*USEFUL\b']
            },
            UtilityToken.SOMEWHAT_USEFUL: {
                'pattern': r'\bSOMEWHAT[_\s]*USEFUL\b',
                'excludes': []
            },
            UtilityToken.NOT_USEFUL: {
                'pattern': r'\b(NOT[_\s]*USEFUL|USELESS)\b',
                'excludes': []
            },
        }

        token, reasoning = self._parse_reflection_token(response['response'], token_patterns)
        return token, reasoning

    def reflect_on_answer(
        self,
        query: str,
        answer: str,
        documents: List[Document],
        parallel: bool = True
    ) -> ReflectionResult:
        """
        Generate all reflection tokens for an answer

        Args:
            query: Original query
            answer: Generated answer
            documents: Retrieved documents
            parallel: Run reflection assessments in parallel (default True)

        Returns:
            ReflectionResult with all assessments
        """
        start_time = time.time()

        if self.verbose:
            logger.info("[REFLECTION] Assessing answer quality..." + (" (parallel)" if parallel else " (sequential)"))

        if parallel:
            # Run all 3 assessments in parallel for ~3x speedup
            relevance = relevance_reasoning = None
            support = support_reasoning = None
            utility = utility_reasoning = None

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all tasks
                future_relevance = executor.submit(self._assess_relevance, query, documents)
                future_support = executor.submit(self._assess_support, answer, documents)
                future_utility = executor.submit(self._assess_utility, query, answer)

                # Collect results as they complete
                futures = {
                    future_relevance: 'relevance',
                    future_support: 'support',
                    future_utility: 'utility'
                }

                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        result = future.result()
                        if task_name == 'relevance':
                            relevance, relevance_reasoning = result
                            if self.verbose:
                                logger.info(f"  Relevance: {relevance.value}")
                        elif task_name == 'support':
                            support, support_reasoning = result
                            if self.verbose:
                                logger.info(f"  Support: {support.value}")
                        elif task_name == 'utility':
                            utility, utility_reasoning = result
                            if self.verbose:
                                logger.info(f"  Utility: {utility.value}")
                    except Exception as e:
                        logger.error(f"  {task_name} assessment failed: {e}")
                        # Set defaults on failure
                        if task_name == 'relevance':
                            relevance, relevance_reasoning = RelevanceToken.PARTIALLY_RELEVANT, f"Error: {e}"
                        elif task_name == 'support':
                            support, support_reasoning = SupportToken.PARTIALLY_SUPPORTED, f"Error: {e}"
                        elif task_name == 'utility':
                            utility, utility_reasoning = UtilityToken.SOMEWHAT_USEFUL, f"Error: {e}"
        else:
            # Sequential execution (original behavior)
            try:
                relevance, relevance_reasoning = self._assess_relevance(query, documents)
            except Exception as e:
                logger.error(f"  Relevance assessment failed: {e}")
                relevance, relevance_reasoning = RelevanceToken.PARTIALLY_RELEVANT, f"Error: {e}"
            if self.verbose:
                logger.info(f"  Relevance: {relevance.value}")

            try:
                support, support_reasoning = self._assess_support(answer, documents)
            except Exception as e:
                logger.error(f"  Support assessment failed: {e}")
                support, support_reasoning = SupportToken.PARTIALLY_SUPPORTED, f"Error: {e}"
            if self.verbose:
                logger.info(f"  Support: {support.value}")

            try:
                utility, utility_reasoning = self._assess_utility(query, answer)
            except Exception as e:
                logger.error(f"  Utility assessment failed: {e}")
                utility, utility_reasoning = UtilityToken.SOMEWHAT_USEFUL, f"Error: {e}"
            if self.verbose:
                logger.info(f"  Utility: {utility.value}")

        reflection_time = time.time() - start_time

        result = ReflectionResult(
            relevance=relevance,
            support=support,
            utility=utility,
            relevance_reasoning=relevance_reasoning,
            support_reasoning=support_reasoning,
            utility_reasoning=utility_reasoning,
            reflection_time=reflection_time
        )

        if self.verbose:
            logger.info(f"  Overall: {result.overall_score:.2f}" + (" [!] Needs improvement" if result.needs_regeneration else ""))

        return result

    def query_with_reflection(
        self,
        query: str,
        documents: List[Document],
        auto_regenerate: bool = True
    ) -> SelfRAGResult:
        """
        Query with automatic reflection and optional regeneration

        Args:
            query: User query
            documents: Retrieved documents
            auto_regenerate: Automatically regenerate if quality is low

        Returns:
            SelfRAGResult with answer and reflection
        """
        total_start = time.time()

        if self.verbose:
            logger.info(f"SELF-RAG QUERY: {query[:60]}... | Documents: {len(documents)}")

        if not documents:
            logger.warning("No documents provided - Self-RAG reflection may produce poor results")

        regeneration_count = 0
        best_answer = None
        best_reflection = None
        best_score = 0.0
        total_generation_time = 0.0

        for attempt in range(self.max_regenerations + 1):
            if self.verbose:
                logger.info(f"[Generation {attempt + 1}]")

            # Generate answer
            answer, gen_time = self._generate_answer(query, documents)
            total_generation_time += gen_time

            # Reflect on answer
            reflection = self.reflect_on_answer(query, answer, documents)

            # Track best answer
            if reflection.overall_score > best_score:
                best_answer = answer
                best_reflection = reflection
                best_score = reflection.overall_score

            # Check if regeneration needed
            if not auto_regenerate or reflection.overall_score >= self.regeneration_threshold:
                break

            if attempt < self.max_regenerations:
                regeneration_count += 1
                if self.verbose:
                    logger.info(f"[!] Score {reflection.overall_score:.2f} below threshold {self.regeneration_threshold}, regenerating...")

        total_time = time.time() - total_start

        result = SelfRAGResult(
            query=query,
            answer=best_answer,
            documents=documents,
            reflection=best_reflection,
            regeneration_count=regeneration_count,
            total_time=total_time,
            generation_time=total_generation_time,
            was_regenerated=regeneration_count > 0
        )

        if self.verbose:
            logger.info(f"SELF-RAG RESULT: score={best_score:.2f}, regenerations={regeneration_count}, time={total_time:.1f}s")

        return result

    def stream_reflection(
        self,
        query: str,
        answer: str,
        documents: List[Document]
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream reflection tokens for UI display

        Args:
            query: Original query
            answer: Generated answer
            documents: Retrieved documents

        Yields:
            Dict with reflection stage and result
        """
        yield {"stage": "start", "message": "Starting reflection assessment..."}

        # Relevance
        yield {"stage": "relevance", "status": "in_progress", "message": "Assessing document relevance..."}
        relevance, relevance_reasoning = self._assess_relevance(query, documents)
        yield {
            "stage": "relevance",
            "status": "complete",
            "token": relevance.value,
            "score": relevance.score,
            "reasoning": relevance_reasoning
        }

        # Support
        yield {"stage": "support", "status": "in_progress", "message": "Assessing answer support..."}
        support, support_reasoning = self._assess_support(answer, documents)
        yield {
            "stage": "support",
            "status": "complete",
            "token": support.value,
            "score": support.score,
            "reasoning": support_reasoning
        }

        # Utility
        yield {"stage": "utility", "status": "in_progress", "message": "Assessing answer utility..."}
        utility, utility_reasoning = self._assess_utility(query, answer)
        yield {
            "stage": "utility",
            "status": "complete",
            "token": utility.value,
            "score": utility.score,
            "reasoning": utility_reasoning
        }

        # Final result
        result = ReflectionResult(
            relevance=relevance,
            support=support,
            utility=utility,
            relevance_reasoning=relevance_reasoning,
            support_reasoning=support_reasoning,
            utility_reasoning=utility_reasoning
        )

        yield {
            "stage": "complete",
            "overall_score": result.overall_score,
            "needs_regeneration": result.needs_regeneration,
            "summary": result.summary()
        }


def test_self_rag():
    """Test Self-RAG functionality"""
    print("=" * 70)
    print("SELF-RAG TEST")
    print("=" * 70)

    try:
        # Initialize
        print("\nInitializing Self-RAG...")
        self_rag = OllamaSelfRAG(verbose=True)
        print("SUCCESS: Self-RAG initialized")

        # Create test documents
        documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
                metadata={"source": "ml_basics"}
            ),
            Document(
                page_content="Deep learning is a type of machine learning that uses neural networks with multiple layers. It excels at tasks like image recognition and natural language processing.",
                metadata={"source": "dl_intro"}
            ),
            Document(
                page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for more accurate and grounded responses.",
                metadata={"source": "rag_overview"}
            )
        ]

        # Test query with reflection
        query = "What is machine learning and how does it relate to deep learning?"

        print(f"\nQuery: {query}")
        print("\nProcessing with Self-RAG...")

        result = self_rag.query_with_reflection(query, documents)

        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Answer: {result.answer[:300]}...")
        print(f"\nReflection Summary: {result.reflection.summary()}")
        print(f"Regeneration count: {result.regeneration_count}")
        print(f"Total time: {result.total_time:.1f}s")

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_self_rag()
