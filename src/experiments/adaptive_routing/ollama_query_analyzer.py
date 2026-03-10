"""
Ollama Query Analyzer
Analyzes query complexity to determine optimal RAG strategy
"""

import time
import logging
import re
from typing import Any, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Import config system
try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple" # 0-3: Direct factual questions
    MEDIUM = "medium" # 4-7: Requires explanation or context
    COMPLEX = "complex" # 8-10: Multi-faceted, comparison, analysis

class QueryCategory(Enum):
    """Query category types for routing"""
    FACTUAL = "factual"
    CREATIVE = "creative"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    RESEARCH = "research"
    COMPARATIVE = "comparative"

@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    query: str
    complexity_score: int # 0-10
    complexity_level: QueryComplexity
    reasoning: str
    confidence: float
    analysis_time: float
    characteristics: Dict[str, Any]
    category: 'QueryCategory' = None  # Query category for routing

class OllamaQueryAnalyzer:
    """
    Analyzes queries to determine complexity and optimal processing strategy

    Uses Ollama to score queries on a 0-10 scale:
    - 0-3: Simple (direct facts, definitions)
    - 4-7: Medium (explanations, how-to)
    - 8-10: Complex (analysis, comparison, multi-step reasoning)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        verbose: Optional[bool] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize query analyzer

        Args:
            model: Ollama model to use
            temperature: Generation temperature (low for consistency)
            verbose: Enable verbose logging
            config: Optional Config object (uses global config if not provided)
            timeout: Timeout for LLM calls in seconds
        """

        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.temperature = temperature if temperature is not None else 0.1  # Low temp for consistent scoring
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.timeout = timeout if timeout is not None else config.llm.timeout
        else:
            # Fallback to hardcoded defaults if no config available
            self.model = model or "qwen2.5:14b"
            self.temperature = temperature if temperature is not None else 0.1
            self.verbose = verbose if verbose is not None else True
            self.timeout = timeout or 120

        # Test connection
        self._test_connection()

        if self.verbose:
            logger.info(f"OllamaQueryAnalyzer initialized with model: {self.model}")

    def _test_connection(self):
        """Test Ollama server connection and verify model availability"""
        try:
            available_models = ollama.list()
            model_names = [m.model for m in available_models.models]

            if self.model not in model_names:
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query complexity

        Args:
            query: User query to analyze

        Returns:
            QueryAnalysis with complexity score and reasoning
        """
        start_time = time.time()

        # Quick heuristic analysis (fast, rough)
        characteristics = self._extract_characteristics(query)

        # LLM-based analysis (slower, accurate)
        llm_score, llm_reasoning = self._llm_complexity_analysis(query)

        # Combine heuristic and LLM scores (weighted average)
        heuristic_score = characteristics['heuristic_score']
        final_score = int(0.3 * heuristic_score + 0.7 * llm_score)

        # Determine complexity level
        if final_score <= 3:
            complexity_level = QueryComplexity.SIMPLE
        elif final_score <= 7:
            complexity_level = QueryComplexity.MEDIUM
        else:
            complexity_level = QueryComplexity.COMPLEX

        # Calculate confidence based on agreement
        score_diff = abs(heuristic_score - llm_score)
        confidence = max(0.5, 1.0 - (score_diff / 10.0))

        analysis_time = time.time() - start_time

        return QueryAnalysis(
            query=query,
            complexity_score=final_score,
            complexity_level=complexity_level,
            reasoning=llm_reasoning,
            confidence=confidence,
            analysis_time=analysis_time,
            characteristics=characteristics
        )

    def _extract_characteristics(self, query: str) -> Dict[str, Any]:
        """
        Extract query characteristics using heuristics

        Args:
            query: User query

        Returns:
            Dictionary of characteristics
        """
        chars = {}

        # Basic metrics
        chars['length'] = len(query)
        chars['word_count'] = len(query.split())
        chars['has_question_mark'] = '?' in query

        # Complexity indicators
        comparison_words = ['compare', 'contrast', 'difference', 'versus', 'vs', 'better', 'worse']
        analysis_words = ['analyze', 'evaluate', 'assess', 'implications', 'impact', 'relationship']
        simple_words = ['what', 'who', 'when', 'where', 'is', 'are', 'define']

        chars['has_comparison'] = any(word in query.lower() for word in comparison_words)
        chars['has_analysis'] = any(word in query.lower() for word in analysis_words)
        chars['is_simple_question'] = any(query.lower().startswith(word) for word in simple_words)

        # Technical depth
        technical_terms = ['algorithm', 'architecture', 'implementation', 'framework', 'paradigm']
        chars['has_technical_terms'] = any(term in query.lower() for term in technical_terms)

        # Heuristic score (0-10)
        score = 5 # Base score

        if chars['is_simple_question']:
            score -= 2
        if chars['word_count'] < 5:
            score -= 1
        if chars['word_count'] > 15:
            score += 1
        if chars['has_comparison']:
            score += 2
        if chars['has_analysis']:
            score += 2
        if chars['has_technical_terms']:
            score += 1

        chars['heuristic_score'] = max(0, min(10, score))

        return chars

    def _llm_complexity_analysis(self, query: str) -> tuple[int, str]:
        """
        Use LLM to analyze query complexity

        Args:
            query: User query

        Returns:
            Tuple of (complexity_score, reasoning)
        """
        prompt = f"""You are a query complexity analyzer. Score the following query on a scale of 0-10:

0-3: SIMPLE - Direct factual questions, definitions, basic "what is" queries
- Examples: "What is machine learning?", "Who invented Python?", "Define RAG"

4-7: MEDIUM - Requires explanation, understanding, or multi-step thinking
- Examples: "How does deep learning work?", "Explain the benefits of RAG systems"

8-10: COMPLEX - Requires analysis, comparison, synthesis, or multi-hop reasoning
- Examples: "Compare quantum computing implications on cryptography vs classical methods", "Analyze the trade-offs between HyDE and baseline RAG across different domains"

Query: "{query}"

Provide:
1. Score (0-10): [number]
2. Brief reasoning (1 sentence)

Format your response as:
Score: [number]
Reasoning: [explanation]"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 150  # Short response
                },
            )
            result = response['response']
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return 5, "Unable to analyze query complexity"

        # Parse response
        score = 5  # Default
        reasoning = result

        # Word-to-number mapping for non-numeric LLM responses
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        }

        try:
            lines = result.strip().split('\n')
            for line in lines:
                if 'score:' in line.lower():
                    # Extract number
                    match = re.search(r'(\d+)', line)
                    if match:
                        score = int(match.group(1))
                        score = max(0, min(10, score))  # Clamp to 0-10
                    else:
                        # Fallback: try word-to-number conversion
                        line_lower = line.lower()
                        for word, num in word_to_num.items():
                            if word in line_lower:
                                score = num
                                break
                        else:
                            logger.warning(f"Could not extract numeric score from: {line.strip()}")
                elif 'reasoning:' in line.lower():
                    reasoning = line.split(':', 1)[1].strip()
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return score, reasoning


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("OLLAMA QUERY ANALYZER TEST")
    print("=" * 70)

    try:
        # Initialize
        analyzer = OllamaQueryAnalyzer(verbose=True)
        print("\nSUCCESS: Analyzer initialized")

        # Test queries at different complexity levels
        test_queries = [
            # Simple (0-3)
            "What is machine learning?",
            "Define RAG",
            "Who created Python?",

            # Medium (4-7)
            "How does deep learning work?",
            "Explain the benefits of HyDE retrieval",

            # Complex (8-10)
            "Compare the implications of quantum computing on modern cryptography",
            "Analyze the trade-offs between HyDE and Self-RAG for different query types"
        ]

        print("\n" + "=" * 70)
        print("ANALYZING QUERIES")
        print("=" * 70)

        results = []
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print("=" * 70)

            analysis = analyzer.analyze_query(query)
            results.append(analysis)

            print(f"Complexity Score: {analysis.complexity_score}/10")
            print(f"Level: {analysis.complexity_level.value.upper()}")
            print(f"Confidence: {analysis.confidence:.2f}")
            print(f"Analysis Time: {analysis.analysis_time:.2f}s")
            print(f"\nCharacteristics:")
            print(f"  - Word count: {analysis.characteristics['word_count']}")
            print(f"  - Has comparison: {analysis.characteristics['has_comparison']}")
            print(f"  - Has analysis: {analysis.characteristics['has_analysis']}")
            print(f"  - Simple question: {analysis.characteristics['is_simple_question']}")
            print(f"\nReasoning: {analysis.reasoning[:150]}...")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        simple_count = sum(1 for r in results if r.complexity_level == QueryComplexity.SIMPLE)
        medium_count = sum(1 for r in results if r.complexity_level == QueryComplexity.MEDIUM)
        complex_count = sum(1 for r in results if r.complexity_level == QueryComplexity.COMPLEX)

        print(f"\nClassification Results:")
        print(f"  Simple: {simple_count}/{len(results)}")
        print(f"  Medium: {medium_count}/{len(results)}")
        print(f"  Complex: {complex_count}/{len(results)}")

        avg_time = sum(r.analysis_time for r in results) / len(results)
        print(f"\nAverage Analysis Time: {avg_time:.2f}s")

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
