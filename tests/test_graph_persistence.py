"""
Test script for GraphRAG persistence (save/load to JSON)
"""

import sys
import os
import tempfile
sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

import pytest
from langchain.schema import Document

try:
    import ollama
    ollama.list()
    OLLAMA_RUNNING = True
except Exception:
    OLLAMA_RUNNING = False


@pytest.mark.skipif(not OLLAMA_RUNNING, reason="Ollama not running")
def test_graph_persistence():
    """Test GraphRAG save and load functionality"""
    from src.experiments.graph_reasoning.ollama_graph_rag import (
        OllamaGraphRAG, Entity, Relationship, Community
    )

    graph_rag = OllamaGraphRAG(verbose=True)

    documents = [
        Document(
            page_content="""Machine learning is a subset of artificial intelligence that enables
            computers to learn from data without being explicitly programmed.""",
            metadata={"source": "ml_basics"}
        ),
        Document(
            page_content="""Deep learning is a type of machine learning that uses neural networks
            with multiple layers. It excels at pattern recognition tasks.""",
            metadata={"source": "dl_intro"}
        ),
    ]

    # Build graph
    stats = graph_rag.build_graph_from_documents(documents)
    original_stats = graph_rag.get_graph_stats()

    # Save graph to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    save_stats = graph_rag.save_graph(temp_path)

    # Clear and reload
    graph_rag.clear_graph()
    empty_stats = graph_rag.get_graph_stats()
    assert empty_stats['entities'] == 0 and empty_stats['nodes'] == 0, "Graph not fully cleared"

    load_stats = graph_rag.load_graph(temp_path)
    restored_stats = graph_rag.get_graph_stats()

    # Cleanup
    os.unlink(temp_path)

    # Verify
    assert original_stats['entities'] == restored_stats['entities'], \
        f"Entities mismatch: {original_stats['entities']} != {restored_stats['entities']}"
    assert original_stats['relationships'] == restored_stats['relationships'], \
        f"Relationships mismatch: {original_stats['relationships']} != {restored_stats['relationships']}"
    assert original_stats['communities'] == restored_stats['communities'], \
        f"Communities mismatch: {original_stats['communities']} != {restored_stats['communities']}"
    assert original_stats['nodes'] == restored_stats['nodes'], \
        f"Graph nodes mismatch: {original_stats['nodes']} != {restored_stats['nodes']}"
    assert original_stats['edges'] == restored_stats['edges'], \
        f"Graph edges mismatch: {original_stats['edges']} != {restored_stats['edges']}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
