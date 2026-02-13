"""
Test script for GraphRAG persistence (save/load to JSON)
"""

import sys
import os
import tempfile
sys.path.insert(0, '/home/dxxc/my_projects/python_projects/adaptive-multimodal-rag')

from langchain.schema import Document


def test_graph_persistence():
    """Test GraphRAG save and load functionality"""
    print("=" * 70)
    print("GRAPHRAG PERSISTENCE TEST")
    print("=" * 70)

    try:
        from src.experiments.graph_reasoning.ollama_graph_rag import (
            OllamaGraphRAG, Entity, Relationship, Community
        )

        # Initialize GraphRAG
        print("\nInitializing GraphRAG...")
        graph_rag = OllamaGraphRAG(verbose=True)
        print("SUCCESS: GraphRAG initialized")

        # Create test documents
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
        print("\n" + "=" * 70)
        print("TEST 1: Build graph from documents")
        print("=" * 70)

        stats = graph_rag.build_graph_from_documents(documents)
        print(f"Build stats: {stats}")

        original_stats = graph_rag.get_graph_stats()
        print(f"Original graph stats: {original_stats}")

        # Save graph to temp file
        print("\n" + "=" * 70)
        print("TEST 2: Save graph to JSON")
        print("=" * 70)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        save_stats = graph_rag.save_graph(temp_path)
        print(f"Save stats: {save_stats}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")

        # Clear graph
        print("\n" + "=" * 70)
        print("TEST 3: Clear graph")
        print("=" * 70)

        graph_rag.clear_graph()
        empty_stats = graph_rag.get_graph_stats()
        print(f"After clear: {empty_stats}")

        if empty_stats['entities'] == 0 and empty_stats['nodes'] == 0:
            print("Graph cleared successfully")
        else:
            print("WARNING: Graph not fully cleared")

        # Load graph
        print("\n" + "=" * 70)
        print("TEST 4: Load graph from JSON")
        print("=" * 70)

        load_stats = graph_rag.load_graph(temp_path)
        print(f"Load stats: {load_stats}")

        restored_stats = graph_rag.get_graph_stats()
        print(f"Restored graph stats: {restored_stats}")

        # Verify
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)

        checks_passed = True

        # Check entities match
        if original_stats['entities'] == restored_stats['entities']:
            print(f"  Entities: PASS ({original_stats['entities']} == {restored_stats['entities']})")
        else:
            print(f"  Entities: FAIL ({original_stats['entities']} != {restored_stats['entities']})")
            checks_passed = False

        # Check relationships match
        if original_stats['relationships'] == restored_stats['relationships']:
            print(f"  Relationships: PASS ({original_stats['relationships']} == {restored_stats['relationships']})")
        else:
            print(f"  Relationships: FAIL ({original_stats['relationships']} != {restored_stats['relationships']})")
            checks_passed = False

        # Check communities match
        if original_stats['communities'] == restored_stats['communities']:
            print(f"  Communities: PASS ({original_stats['communities']} == {restored_stats['communities']})")
        else:
            print(f"  Communities: FAIL ({original_stats['communities']} != {restored_stats['communities']})")
            checks_passed = False

        # Check graph nodes match
        if original_stats['nodes'] == restored_stats['nodes']:
            print(f"  Graph nodes: PASS ({original_stats['nodes']} == {restored_stats['nodes']})")
        else:
            print(f"  Graph nodes: FAIL ({original_stats['nodes']} != {restored_stats['nodes']})")
            checks_passed = False

        # Check graph edges match
        if original_stats['edges'] == restored_stats['edges']:
            print(f"  Graph edges: PASS ({original_stats['edges']} == {restored_stats['edges']})")
        else:
            print(f"  Graph edges: FAIL ({original_stats['edges']} != {restored_stats['edges']})")
            checks_passed = False

        # Cleanup
        os.unlink(temp_path)

        print("\n" + "=" * 70)
        if checks_passed:
            print("ALL TESTS PASSED! Graph persistence working correctly.")
        else:
            print("SOME TESTS FAILED")
        print("=" * 70)

        return checks_passed

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_graph_persistence()
    sys.exit(0 if success else 1)
