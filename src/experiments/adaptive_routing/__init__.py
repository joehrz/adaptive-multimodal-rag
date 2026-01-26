"""
Adaptive RAG Routing: Intelligent Query Routing System

This module implements an intelligent query routing system that:
1. Analyzes query complexity and category
2. Selects optimal RAG strategies based on analysis
3. Executes strategies with fallback handling
4. Tracks performance and learns from results
5. Provides business intelligence and cost optimization
"""

from .ollama_query_analyzer import OllamaQueryAnalyzer, QueryAnalysis, QueryComplexity
from .ollama_router import OllamaAdaptiveRouter, RAGStrategy, RoutingDecision

__all__ = [
    'OllamaQueryAnalyzer',
    'QueryAnalysis',
    'QueryComplexity',
    'OllamaAdaptiveRouter',
    'RAGStrategy',
    'RoutingDecision'
]