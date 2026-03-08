"""
Tests for Adaptive Routing: RAGStrategy enum, OllamaAdaptiveRouter, and OllamaQueryAnalyzer.
All Ollama/LLM calls are mocked - no running Ollama required.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from src.experiments.adaptive_routing.ollama_router import (
    RAGStrategy,
    RoutingDecision,
    StrategyResult,
    OllamaAdaptiveRouter,
)
from src.experiments.adaptive_routing.ollama_query_analyzer import (
    QueryComplexity,
    QueryCategory,
    QueryAnalysis,
    OllamaQueryAnalyzer,
)


# --- Enum tests ---

class TestRAGStrategy:
    def test_all_strategies(self):
        assert RAGStrategy.BASELINE.value == "baseline"
        assert RAGStrategy.HYDE.value == "hyde"
        assert RAGStrategy.SELF_RAG.value == "self_rag"
        assert RAGStrategy.HYDE_SELF_RAG.value == "hyde_self_rag"
        assert RAGStrategy.MULTIMODAL.value == "multimodal"
        assert RAGStrategy.GRAPHRAG.value == "graphrag"

    def test_strategy_count(self):
        assert len(RAGStrategy) == 6


class TestQueryComplexity:
    def test_levels(self):
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MEDIUM.value == "medium"
        assert QueryComplexity.COMPLEX.value == "complex"


class TestQueryCategory:
    def test_categories(self):
        assert QueryCategory.FACTUAL.value == "factual"
        assert QueryCategory.ANALYTICAL.value == "analytical"
        assert QueryCategory.COMPARATIVE.value == "comparative"


# --- QueryAnalysis dataclass ---

class TestQueryAnalysis:
    def test_creation(self):
        analysis = QueryAnalysis(
            query="test",
            complexity_score=5,
            complexity_level=QueryComplexity.MEDIUM,
            reasoning="medium complexity",
            confidence=0.8,
            analysis_time=0.1,
            characteristics={"word_count": 1},
        )
        assert analysis.complexity_score == 5
        assert analysis.category is None  # default


# --- OllamaQueryAnalyzer tests (mocked) ---

@pytest.fixture
def mock_ollama_for_analyzer():
    mock_model = MagicMock()
    mock_model.model = "qwen2.5:14b"
    mock_list_result = MagicMock()
    mock_list_result.models = [mock_model]

    with patch('src.experiments.adaptive_routing.ollama_query_analyzer.ollama') as mock_ol, \
         patch('src.experiments.adaptive_routing.ollama_query_analyzer.OLLAMA_AVAILABLE', True), \
         patch('src.experiments.adaptive_routing.ollama_query_analyzer.CONFIG_AVAILABLE', False):
        mock_ol.list.return_value = mock_list_result
        mock_ol.generate.return_value = {'response': 'Score: 5\nReasoning: Medium complexity query'}
        yield mock_ol


@pytest.fixture
def analyzer(mock_ollama_for_analyzer):
    return OllamaQueryAnalyzer(model="qwen2.5:14b", verbose=False)


class TestQueryAnalyzerInit:
    def test_init_raises_without_ollama(self):
        with patch('src.experiments.adaptive_routing.ollama_query_analyzer.OLLAMA_AVAILABLE', False), \
             patch('src.experiments.adaptive_routing.ollama_query_analyzer.CONFIG_AVAILABLE', False):
            with pytest.raises(ImportError, match="ollama package not found"):
                OllamaQueryAnalyzer()

    def test_init_with_defaults(self, analyzer):
        assert analyzer.model == "qwen2.5:14b"
        assert analyzer.temperature == 0.1


class TestExtractCharacteristics:
    def test_simple_question(self, analyzer):
        chars = analyzer._extract_characteristics("What is Python?")
        assert chars['is_simple_question'] is True
        assert chars['has_question_mark'] is True
        assert chars['heuristic_score'] <= 4

    def test_comparison_question(self, analyzer):
        chars = analyzer._extract_characteristics("Compare transformers versus RNNs for NLP tasks")
        assert chars['has_comparison'] is True
        assert chars['heuristic_score'] >= 5

    def test_analysis_question(self, analyzer):
        chars = analyzer._extract_characteristics("Analyze the implications of the framework paradigm")
        assert chars['has_analysis'] is True
        assert chars['has_technical_terms'] is True
        assert chars['heuristic_score'] >= 7

    def test_short_query(self, analyzer):
        chars = analyzer._extract_characteristics("Hi")
        assert chars['word_count'] < 5
        assert chars['heuristic_score'] <= 5

    def test_long_query(self, analyzer):
        chars = analyzer._extract_characteristics(
            "Can you explain in detail how the transformer architecture works and why it is better than recurrent neural networks for natural language processing tasks"
        )
        assert chars['word_count'] > 15

    def test_score_clamped(self, analyzer):
        # Even extreme queries should be clamped 0-10
        chars = analyzer._extract_characteristics("What is it?")
        assert 0 <= chars['heuristic_score'] <= 10

        chars = analyzer._extract_characteristics(
            "Analyze and compare the implications of the algorithm framework paradigm architecture implementation"
        )
        assert 0 <= chars['heuristic_score'] <= 10


class TestLLMComplexityAnalysis:
    def test_parses_score_and_reasoning(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'Score: 7\nReasoning: Requires multi-step explanation'
        }
        score, reasoning = analyzer._llm_complexity_analysis("How does deep learning work?")
        assert score == 7
        assert "multi-step" in reasoning.lower()

    def test_handles_malformed_response(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'This is not a valid format at all'
        }
        score, reasoning = analyzer._llm_complexity_analysis("test")
        assert score == 5  # default
        assert isinstance(reasoning, str)

    def test_clamps_score(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'Score: 15\nReasoning: Very complex'
        }
        score, _ = analyzer._llm_complexity_analysis("test")
        assert score == 10

    def test_handles_ollama_error(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.side_effect = Exception("connection error")
        score, reasoning = analyzer._llm_complexity_analysis("test")
        assert score == 5
        assert "unable" in reasoning.lower()


class TestAnalyzeQuery:
    def test_simple_query(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'Score: 2\nReasoning: Simple factual question'
        }
        result = analyzer.analyze_query("What is Python?")
        assert result.complexity_level == QueryComplexity.SIMPLE
        assert result.complexity_score <= 3

    def test_complex_query(self, analyzer, mock_ollama_for_analyzer):
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'Score: 9\nReasoning: Complex analytical query'
        }
        result = analyzer.analyze_query("Analyze the trade-offs between HyDE and Self-RAG")
        assert result.complexity_level == QueryComplexity.COMPLEX
        assert result.complexity_score >= 8

    def test_confidence_high_when_scores_agree(self, analyzer, mock_ollama_for_analyzer):
        # Heuristic for "What is Python?" is low; make LLM agree
        mock_ollama_for_analyzer.generate.return_value = {
            'response': 'Score: 2\nReasoning: Simple'
        }
        result = analyzer.analyze_query("What is Python?")
        assert result.confidence >= 0.7


# --- OllamaAdaptiveRouter tests (mocked) ---

@pytest.fixture
def mock_analyzer():
    """Create a mock query analyzer that returns configurable results."""
    analyzer = MagicMock(spec=OllamaQueryAnalyzer)
    return analyzer


def _make_analysis(score, level=None):
    if level is None:
        if score <= 3:
            level = QueryComplexity.SIMPLE
        elif score <= 7:
            level = QueryComplexity.MEDIUM
        else:
            level = QueryComplexity.COMPLEX
    return QueryAnalysis(
        query="test",
        complexity_score=score,
        complexity_level=level,
        reasoning="test reason",
        confidence=0.9,
        analysis_time=0.01,
        characteristics={"heuristic_score": score},
    )


class TestRouterRouting:
    def test_simple_query_routes_to_baseline(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(2)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("What is Python?")
        assert decision.selected_strategy == RAGStrategy.BASELINE

    def test_medium_query_routes_to_hyde(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(5)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("How does deep learning work?")
        assert decision.selected_strategy == RAGStrategy.HYDE

    def test_complex_query_routes_to_hyde_self_rag(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(9)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Analyze the trade-offs between approaches")
        assert decision.selected_strategy == RAGStrategy.HYDE_SELF_RAG

    def test_visual_keyword_routes_to_multimodal(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(5)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Describe the image in figure 3")
        assert decision.selected_strategy == RAGStrategy.MULTIMODAL

    def test_visual_keywords_comprehensive(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(5)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        for keyword in ['image', 'figure', 'chart', 'diagram', 'table', 'plot', 'photo']:
            decision = router.route_query(f"Show the {keyword}")
            assert decision.selected_strategy == RAGStrategy.MULTIMODAL, f"Failed for keyword: {keyword}"

    def test_graphrag_for_relationship_queries(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(7)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("What is the relationship between transformers and attention?")
        assert decision.selected_strategy == RAGStrategy.GRAPHRAG

    def test_graphrag_requires_high_complexity(self, mock_analyzer):
        # "relationship between" but low complexity should NOT route to graphrag
        mock_analyzer.analyze_query.return_value = _make_analysis(3)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("What is the relationship between A and B?")
        assert decision.selected_strategy == RAGStrategy.BASELINE

    def test_summarization_never_routes_to_graphrag(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(8)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Summarize the relationship between the main points")
        assert decision.selected_strategy != RAGStrategy.GRAPHRAG

    def test_summarization_routes_to_hyde_not_baseline(self, mock_analyzer):
        """Summarization queries should never route to BASELINE even with low complexity score"""
        mock_analyzer.analyze_query.return_value = _make_analysis(1)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Can you give me a summary of this paper?")
        assert decision.selected_strategy == RAGStrategy.HYDE

    def test_summarization_overview_routes_to_hyde(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(2)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Give me an overview of the key points")
        assert decision.selected_strategy == RAGStrategy.HYDE

    def test_complex_summarization_routes_to_hyde_self_rag(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(9)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Summarize and critically evaluate the methodology")
        assert decision.selected_strategy == RAGStrategy.HYDE_SELF_RAG

    def test_multimodal_takes_priority_over_graphrag(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(8)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("Show the figure about the relationship between A and B")
        assert decision.selected_strategy == RAGStrategy.MULTIMODAL


class TestRouterDecisionFields:
    def test_decision_has_all_fields(self, mock_analyzer):
        mock_analyzer.analyze_query.return_value = _make_analysis(5)
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        decision = router.route_query("test query")
        assert isinstance(decision, RoutingDecision)
        assert decision.query == "test query"
        assert isinstance(decision.complexity_score, int)
        assert isinstance(decision.complexity_level, QueryComplexity)
        assert isinstance(decision.selected_strategy, RAGStrategy)
        assert isinstance(decision.reasoning, str)
        assert isinstance(decision.expected_latency, float)
        assert isinstance(decision.expected_quality, float)
        assert decision.routing_time >= 0


class TestRouterStats:
    def test_initial_stats(self, mock_analyzer):
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        stats = router.get_stats()
        assert stats['total_queries_routed'] == 0

    def test_update_stats(self, mock_analyzer):
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        router._update_stats(RAGStrategy.BASELINE, 2.0, 0.8)
        assert router.strategy_stats[RAGStrategy.BASELINE]['count'] == 1

    def test_running_average(self, mock_analyzer):
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        router._update_stats(RAGStrategy.BASELINE, 4.0, 0.9)
        router._update_stats(RAGStrategy.BASELINE, 6.0, 0.7)
        stats = router.strategy_stats[RAGStrategy.BASELINE]
        assert stats['count'] == 2
        # Running average from initial (3.0, 0.75) -> after (4.0, 0.9) -> after (6.0, 0.7)
        # First update: avg_latency = (3.0*0 + 4.0)/1 = 4.0
        # Second update: avg_latency = (4.0*1 + 6.0)/2 = 5.0
        assert stats['avg_latency'] == 5.0


class TestRouterPatternMatch:
    def test_simple_pattern(self, mock_analyzer):
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        assert router._pattern_match("relationship between", "the relationship between A and B")
        assert not router._pattern_match("relationship between", "just a normal query")

    def test_wildcard_pattern(self, mock_analyzer):
        router = OllamaAdaptiveRouter(query_analyzer=mock_analyzer, verbose=False)
        assert router._pattern_match("how does .* relate to", "how does X relate to Y")
        assert router._pattern_match("compare .* with", "compare A with B")
