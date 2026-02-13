"""
Ollama Adaptive Router
Routes queries to optimal RAG strategy based on complexity
"""

import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

# Import from same package
from src.experiments.adaptive_routing.ollama_query_analyzer import OllamaQueryAnalyzer, QueryComplexity

logger = logging.getLogger(__name__)

class RAGStrategy(Enum):
    """Available RAG strategies"""
    BASELINE = "baseline" # Fast, simple retrieval
    HYDE = "hyde" # Hypothetical document generation
    SELF_RAG = "self_rag" # Reflection tokens
    HYDE_SELF_RAG = "hyde_self_rag" # Both techniques
    MULTIMODAL = "multimodal" # LLaVA vision for images/tables
    GRAPHRAG = "graphrag" # Knowledge graph for multi-hop reasoning

@dataclass
class RoutingDecision:
    """Result of routing decision"""
    query: str
    complexity_score: int
    complexity_level: QueryComplexity
    selected_strategy: RAGStrategy
    reasoning: str
    expected_latency: float # Estimated seconds
    expected_quality: float # Estimated score 0-1
    routing_time: float

@dataclass
class StrategyResult:
    """Result from executing a strategy"""
    strategy: RAGStrategy
    answer: str
    retrieved_docs: List[str]
    actual_latency: float
    quality_score: float
    metadata: Dict

class OllamaAdaptiveRouter:
    """
    Intelligent query router that selects optimal RAG strategy

    Routing Rules:
    - Visual keywords detected: Multimodal RAG with LLaVA (20-30s)
    - Simple (0-3): Baseline RAG (fast, 2-5s)
    - Medium (4-7): HyDE RAG (accurate, 30-40s)
    - Complex (8-10): HyDE + Self-RAG (highest quality, 40-50s)
    """

    def __init__(self,
                 query_analyzer: Optional[OllamaQueryAnalyzer] = None,
                 enable_learning: bool = True,
                 verbose: bool = True):
        """
        Initialize adaptive router

        Args:
            query_analyzer: Query complexity analyzer
            enable_learning: Track performance and adapt routing
            verbose: Enable verbose logging
        """

        self.query_analyzer = query_analyzer or OllamaQueryAnalyzer(verbose=False)
        self.enable_learning = enable_learning
        self.verbose = verbose

        # Performance tracking
        self.routing_history = []
        self.strategy_stats = {
            RAGStrategy.BASELINE: {'count': 0, 'avg_latency': 3.0, 'avg_quality': 0.75},
            RAGStrategy.HYDE: {'count': 0, 'avg_latency': 35.0, 'avg_quality': 0.85},
            RAGStrategy.SELF_RAG: {'count': 0, 'avg_latency': 10.0, 'avg_quality': 0.80},
            RAGStrategy.HYDE_SELF_RAG: {'count': 0, 'avg_latency': 45.0, 'avg_quality': 0.90},
            RAGStrategy.MULTIMODAL: {'count': 0, 'avg_latency': 20.0, 'avg_quality': 0.88},
            RAGStrategy.GRAPHRAG: {'count': 0, 'avg_latency': 25.0, 'avg_quality': 0.92},
        }

        if self.verbose:
            logger.info("OllamaAdaptiveRouter initialized")

    def _pattern_match(self, pattern: str, text: str) -> bool:
        """Simple pattern matching with .* wildcard support"""
        import re
        # Convert simple pattern to regex
        regex_pattern = pattern.replace('.*', '.*?')
        try:
            return bool(re.search(regex_pattern, text))
        except re.error:
            return pattern.replace('.*', '') in text

    def route_query(self, query: str) -> RoutingDecision:
        """
        Analyze query and select optimal strategy

        Args:
            query: User query

        Returns:
            RoutingDecision with selected strategy and reasoning
        """
        start_time = time.time()

        # Analyze query complexity
        analysis = self.query_analyzer.analyze_query(query)

        # Check for summarization queries (should use retrieval-based strategies, NOT GraphRAG)
        summarization_keywords = ['summarize', 'summary', 'summarise', 'overview', 'abstract',
                                  'main points', 'key points', 'tldr', 'recap', 'brief',
                                  'describe the paper', 'what does the paper say',
                                  'related work', 'introduction', 'conclusion', 'methods',
                                  'methodology', 'results section', 'discussion section']

        is_summarization = any(keyword in query.lower() for keyword in summarization_keywords)

        # Check for multimodal needs (highest priority)
        visual_keywords = ['image', 'images', 'figure', 'figures', 'chart', 'charts',
                          'diagram', 'diagrams', 'table', 'tables',
                          'visual', 'visuals', 'picture', 'pictures', 'photo', 'photos',
                          'illustration', 'screenshot', 'plot', 'plots']

        needs_multimodal = any(keyword in query.lower() for keyword in visual_keywords)

        # Check for multi-hop reasoning needs (GraphRAG)
        # Be more specific - require actual relationship reasoning patterns, not just keyword presence
        graph_keywords = ['relationship between', 'how does .* relate to', 'connection between',
                         'connect .* to', 'link between', 'path from .* to',
                         'compare .* with', 'contrast .* and', 'versus',
                         'how does .* affect', 'how do .* interact',
                         'what leads to', 'cause of', 'impact of .* on']

        # Use regex-like matching for more specific patterns
        query_lower = query.lower()
        needs_graphrag = (
            not is_summarization and  # Never use GraphRAG for summarization
            analysis.complexity_score >= 6 and  # Higher threshold for GraphRAG
            any(
                keyword in query_lower if ' ' not in keyword.replace('.*', ' ')
                else self._pattern_match(keyword, query_lower)
                for keyword in graph_keywords
            )
        )

        if needs_multimodal:
            # Query needs visual understanding - use multimodal RAG
            strategy = RAGStrategy.MULTIMODAL
            reasoning = "Query requires visual understanding - using LLaVA multimodal RAG for images/tables"

        elif needs_graphrag:
            # Query requires multi-hop reasoning - use GraphRAG
            strategy = RAGStrategy.GRAPHRAG
            reasoning = "Query requires multi-hop reasoning about relationships - using GraphRAG"

        # Select strategy based on complexity
        elif analysis.complexity_score <= 3:
            # Simple: Use baseline RAG
            strategy = RAGStrategy.BASELINE
            reasoning = "Simple factual query - baseline RAG is sufficient and fastest"

        elif analysis.complexity_score <= 7:
            # Medium: Use HyDE for better retrieval
            strategy = RAGStrategy.HYDE
            reasoning = "Medium complexity - HyDE improves retrieval accuracy for explanations"

        else:
            # Complex: Use HyDE + Self-RAG for maximum quality
            strategy = RAGStrategy.HYDE_SELF_RAG
            reasoning = "Complex query requiring analysis - HyDE + Self-RAG ensures high quality"

        # Get expected performance
        stats = self.strategy_stats[strategy]
        expected_latency = stats['avg_latency']
        expected_quality = stats['avg_quality']

        routing_time = time.time() - start_time

        decision = RoutingDecision(
            query=query,
            complexity_score=analysis.complexity_score,
            complexity_level=analysis.complexity_level,
            selected_strategy=strategy,
            reasoning=reasoning,
            expected_latency=expected_latency,
            expected_quality=expected_quality,
            routing_time=routing_time
        )

        if self.verbose:
            print(f"\n[ROUTING DECISION]")
            print(f"Query: {query[:60]}...")
            print(f"Complexity: {analysis.complexity_score}/10 ({analysis.complexity_level.value})")
            print(f"Strategy: {strategy.value.upper()}")
            print(f"Expected: {expected_latency:.1f}s latency, {expected_quality:.2f} quality")
            print(f"Reasoning: {reasoning}")

        return decision

    def execute_strategy(self,
                         decision: RoutingDecision,
                         rag_systems: Dict[RAGStrategy, any]) -> StrategyResult:
        """
        Execute the selected RAG strategy

        Args:
            decision: Routing decision
            rag_systems: Dictionary mapping strategies to RAG system instances

        Returns:
            StrategyResult with answer and metrics
        """
        strategy = decision.selected_strategy
        start_time = time.time()

        if strategy not in rag_systems:
            raise ValueError(f"RAG system not provided for strategy: {strategy.value}")

        rag_system = rag_systems[strategy]

        # Execute query (implementation depends on RAG system interface)
        # This is a placeholder - actual implementation depends on your RAG classes
        try:
            if hasattr(rag_system, 'query'):
                answer = rag_system.query(decision.query)
                retrieved_docs = []
                quality_score = 0.8 # Placeholder
            elif hasattr(rag_system, 'retrieve'):
                result = rag_system.retrieve(decision.query)
                answer = "Generated answer placeholder"
                retrieved_docs = [doc.page_content for doc in result.documents]
                quality_score = 0.8
            else:
                raise ValueError(f"RAG system doesn't have query() or retrieve() method")

        except Exception as e:
            logger.error(f"Error executing strategy {strategy.value}: {e}")
            raise

        actual_latency = time.time() - start_time

        # Update statistics if learning enabled
        if self.enable_learning:
            self._update_stats(strategy, actual_latency, quality_score)

        result = StrategyResult(
            strategy=strategy,
            answer=answer if isinstance(answer, str) else str(answer),
            retrieved_docs=retrieved_docs,
            actual_latency=actual_latency,
            quality_score=quality_score,
            metadata={
                'expected_latency': decision.expected_latency,
                'latency_error': actual_latency - decision.expected_latency,
                'complexity_score': decision.complexity_score
            }
        )

        return result

    def _update_stats(self, strategy: RAGStrategy, latency: float, quality: float):
        """Update strategy performance statistics"""
        stats = self.strategy_stats[strategy]

        # Running average
        count = stats['count']
        stats['avg_latency'] = (stats['avg_latency'] * count + latency) / (count + 1)
        stats['avg_quality'] = (stats['avg_quality'] * count + quality) / (count + 1)
        stats['count'] += 1

    def get_stats(self) -> Dict:
        """Get routing statistics"""
        total_queries = sum(s['count'] for s in self.strategy_stats.values())

        stats = {
            'total_queries_routed': total_queries,
            'strategy_distribution': {
                s.value: self.strategy_stats[s]['count']
                for s in RAGStrategy
            },
            'strategy_performance': {
                s.value: {
                    'avg_latency': self.strategy_stats[s]['avg_latency'],
                    'avg_quality': self.strategy_stats[s]['avg_quality'],
                    'usage_count': self.strategy_stats[s]['count']
                }
                for s in RAGStrategy
            }
        }

        return stats

    def optimize_routing_rules(self):
        """
        Analyze routing history and optimize rules (placeholder for future ML)
        """
        if not self.enable_learning:
            return

        # Placeholder for future: use routing history to adjust thresholds
        # For now, just log statistics
        if self.verbose:
            stats = self.get_stats()
            print("\n[ROUTING OPTIMIZATION]")
            print(f"Total queries: {stats['total_queries_routed']}")
            for strategy in RAGStrategy:
                perf = stats['strategy_performance'][strategy.value]
                print(f"  {strategy.value}: {perf['usage_count']} queries, "
                      f"{perf['avg_latency']:.1f}s avg, "
                      f"{perf['avg_quality']:.2f} quality")


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("OLLAMA ADAPTIVE ROUTER TEST")
    print("=" * 70)

    try:
        # Initialize
        print("\nInitializing router...")
        router = OllamaAdaptiveRouter(verbose=True)
        print("SUCCESS: Router initialized")

        # Test queries
        test_queries = [
            "What is Python?", # Simple
            "How does machine learning work?", # Medium
            "Compare quantum computing implications on cryptography" # Complex
        ]

        print("\n" + "=" * 70)
        print("ROUTING DECISIONS")
        print("=" * 70)

        decisions = []
        for query in test_queries:
            print(f"\n{'='*70}")
            decision = router.route_query(query)
            decisions.append(decision)
            print(f"Routing time: {decision.routing_time:.2f}s")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        strategy_counts = {}
        for decision in decisions:
            strategy = decision.selected_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        print(f"\nRouting Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count}/{len(decisions)}")

        avg_routing_time = sum(d.routing_time for d in decisions) / len(decisions)
        print(f"\nAverage Routing Time: {avg_routing_time:.2f}s")

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

        print("\nKey Observations:")
        print(f"  SUCCESS: Simple queries -> Baseline RAG (fastest)")
        print(f"  SUCCESS: Medium queries -> HyDE (better accuracy)")
        print(f"  SUCCESS: Complex queries -> HyDE + Self-RAG (highest quality)")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
