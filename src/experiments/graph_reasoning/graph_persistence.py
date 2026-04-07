"""Graph persistence: save and load knowledge graphs to/from JSON"""

import time
import json
import logging
from typing import Dict, List, Any

import networkx as nx
from langchain.schema import Document

from src.experiments.graph_reasoning.models import Entity, Relationship, Community

logger = logging.getLogger(__name__)


def save_graph(
    file_path: str,
    graph: nx.DiGraph,
    entities: Dict[str, Entity],
    relationships: List[Relationship],
    communities: Dict[str, Community],
    documents: Dict[str, Document],
    get_graph_stats: callable,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Save the knowledge graph to a JSON file

    Args:
        file_path: Path to save the graph (should end in .json)
        graph: The NetworkX directed graph
        entities: Entity storage dict
        relationships: Relationship list
        communities: Community storage dict
        documents: Document storage dict
        get_graph_stats: Callable that returns graph statistics
        verbose: Whether to log progress

    Returns:
        Dictionary with save statistics
    """
    from pathlib import Path

    start_time = time.time()

    # Serialize entities
    entities_data = {}
    for entity_id, entity in entities.items():
        entities_data[entity_id] = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "source_docs": entity.source_docs,
            "attributes": entity.attributes
        }

    # Serialize relationships
    relationships_data = []
    for rel in relationships:
        relationships_data.append({
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "relation_type": rel.relation_type,
            "description": rel.description,
            "weight": rel.weight,
            "source_doc": rel.source_doc
        })

    # Serialize communities
    communities_data = {}
    for comm_id, community in communities.items():
        communities_data[comm_id] = {
            "id": community.id,
            "entities": community.entities,
            "summary": community.summary,
            "central_entity": community.central_entity,
            "level": community.level
        }

    # Serialize documents (metadata only, content is often too large)
    documents_data = {}
    for doc_id, doc in documents.items():
        documents_data[doc_id] = {
            "content": doc.page_content,
            "metadata": doc.metadata
        }

    # Build graph edges from NetworkX graph
    graph_edges = []
    for source, target, data in graph.edges(data=True):
        graph_edges.append({
            "source": source,
            "target": target,
            "data": data
        })

    # Combine all data
    graph_data = {
        "version": "1.0",
        "entities": entities_data,
        "relationships": relationships_data,
        "communities": communities_data,
        "documents": documents_data,
        "graph_edges": graph_edges,
        "stats": get_graph_stats()
    }

    # Save to file
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    save_time = time.time() - start_time

    stats = {
        "file_path": file_path,
        "entities_saved": len(entities_data),
        "relationships_saved": len(relationships_data),
        "communities_saved": len(communities_data),
        "documents_saved": len(documents_data),
        "save_time": save_time
    }

    if verbose:
        logger.info(f"Graph saved to {file_path}: {stats['entities_saved']} entities, {stats['relationships_saved']} relationships in {save_time:.2f}s")

    return stats


def load_graph(
    file_path: str,
    graph: nx.DiGraph,
    entities: Dict[str, Entity],
    relationships: List[Relationship],
    communities: Dict[str, Community],
    documents: Dict[str, Document],
    clear_graph_fn: callable,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Load the knowledge graph from a JSON file

    Args:
        file_path: Path to the saved graph JSON file
        graph: The NetworkX directed graph to populate
        entities: Entity storage dict to populate
        relationships: Relationship list to populate
        communities: Community storage dict to populate
        documents: Document storage dict to populate
        clear_graph_fn: Callable that clears all graph state
        verbose: Whether to log progress

    Returns:
        Dictionary with load statistics
    """
    from pathlib import Path

    start_time = time.time()

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Graph file not found: {file_path}")

    # Load from file
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Clear existing graph
    clear_graph_fn()

    # Restore entities
    for entity_id, entity_data in graph_data.get("entities", {}).items():
        entity = Entity(
            id=entity_data["id"],
            name=entity_data["name"],
            entity_type=entity_data["entity_type"],
            description=entity_data.get("description", ""),
            source_docs=entity_data.get("source_docs", []),
            attributes=entity_data.get("attributes", {})
        )
        entities[entity_id] = entity
        graph.add_node(
            entity_id,
            name=entity.name,
            type=entity.entity_type,
            description=entity.description
        )

    # Restore relationships
    for rel_data in graph_data.get("relationships", []):
        relationship = Relationship(
            source_id=rel_data["source_id"],
            target_id=rel_data["target_id"],
            relation_type=rel_data["relation_type"],
            description=rel_data.get("description", ""),
            weight=rel_data.get("weight", 1.0),
            source_doc=rel_data.get("source_doc", "")
        )
        relationships.append(relationship)

    # Restore graph edges
    for edge_data in graph_data.get("graph_edges", []):
        graph.add_edge(
            edge_data["source"],
            edge_data["target"],
            **edge_data.get("data", {})
        )

    # Restore communities
    for comm_id, comm_data in graph_data.get("communities", {}).items():
        community = Community(
            id=comm_data["id"],
            entities=comm_data["entities"],
            summary=comm_data.get("summary", ""),
            central_entity=comm_data.get("central_entity"),
            level=comm_data.get("level", 0)
        )
        communities[comm_id] = community

    # Restore documents
    for doc_id, doc_data in graph_data.get("documents", {}).items():
        doc = Document(
            page_content=doc_data["content"],
            metadata=doc_data.get("metadata", {})
        )
        documents[doc_id] = doc

    load_time = time.time() - start_time

    stats = {
        "file_path": file_path,
        "entities_loaded": len(entities),
        "relationships_loaded": len(relationships),
        "communities_loaded": len(communities),
        "documents_loaded": len(documents),
        "load_time": load_time
    }

    if verbose:
        logger.info(f"Graph loaded from {file_path}: {stats['entities_loaded']} entities, {stats['relationships_loaded']} relationships in {load_time:.2f}s")

    return stats
