"""
GraphRAG Implementation with Ollama
Builds knowledge graphs from documents for multi-hop reasoning

Based on "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"

Features:
- Entity extraction from documents
- Relationship identification
- Community detection (Louvain algorithm)
- Multi-hop graph traversal for complex queries
"""

import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

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


@dataclass
class Entity:
    """Extracted entity from documents"""
    id: str
    name: str
    entity_type: str  # e.g., "CONCEPT", "PERSON", "TECHNOLOGY", "ORGANIZATION"
    description: str = ""
    source_docs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relationship:
    """Relationship/edge between entities"""
    source_id: str
    target_id: str
    relation_type: str  # e.g., "IS_A", "RELATES_TO", "USES", "PART_OF"
    description: str = ""
    weight: float = 1.0
    source_doc: str = ""

    @property
    def id(self) -> str:
        return f"{self.source_id}_{self.relation_type}_{self.target_id}"


@dataclass
class Community:
    """Community of related entities"""
    id: str
    entities: List[str]  # Entity IDs
    summary: str = ""
    central_entity: Optional[str] = None
    level: int = 0  # Hierarchy level


@dataclass
class GraphRAGResult:
    """Result from GraphRAG query"""
    query: str
    answer: str
    reasoning_path: List[Dict[str, Any]]  # Steps in the reasoning
    entities_used: List[str]
    relationships_used: List[str]
    communities_consulted: List[str]
    total_time: float = 0.0
    num_hops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "reasoning_path": self.reasoning_path,
            "entities_used": self.entities_used,
            "relationships_used": self.relationships_used,
            "communities_consulted": self.communities_consulted,
            "total_time": self.total_time,
            "num_hops": self.num_hops
        }


class OllamaGraphRAG:
    """
    GraphRAG implementation using Ollama and NetworkX

    Key features:
    - LLM-based entity and relationship extraction
    - Community detection for hierarchical summarization
    - Multi-hop reasoning for complex queries
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_hops: Optional[int] = None,
        verbose: Optional[bool] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        max_entities_per_doc: Optional[int] = None,
        max_relationships_per_doc: Optional[int] = None,
    ):
        """
        Initialize GraphRAG system

        Args:
            model: Ollama model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens for generation
            max_hops: Maximum hops in graph traversal
            verbose: Enable verbose logging
            config: Optional Config object (uses global config if not provided)
            timeout: Timeout for LLM calls in seconds
            max_entities_per_doc: Maximum entities to extract per document
            max_relationships_per_doc: Maximum relationships to extract per document
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx package not found. Install with: pip install networkx")

        # Load config - use provided config or global config
        if config is None and CONFIG_AVAILABLE:
            config = get_config()

        # Apply config defaults, then override with explicit parameters
        if config:
            self.model = model if model is not None else config.llm.model
            self.temperature = temperature if temperature is not None else config.llm.temperature
            self.max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens
            self.max_hops = max_hops if max_hops is not None else config.strategies.graphrag.max_hops
            self.verbose = verbose if verbose is not None else config.logging.verbose
            self.timeout = timeout if timeout is not None else config.strategies.graphrag.timeout
            self.max_entities_per_doc = max_entities_per_doc if max_entities_per_doc is not None else config.strategies.graphrag.max_entities_per_doc
            self.max_relationships_per_doc = max_relationships_per_doc if max_relationships_per_doc is not None else config.strategies.graphrag.max_relationships_per_doc
        else:
            # Fallback to hardcoded defaults if no config available
            self.model = model or "qwen2.5:14b"
            self.temperature = temperature if temperature is not None else 0.3
            self.max_tokens = max_tokens or 1000
            self.max_hops = max_hops or 3
            self.verbose = verbose if verbose is not None else True
            self.timeout = timeout or 60
            self.max_entities_per_doc = max_entities_per_doc or 7
            self.max_relationships_per_doc = max_relationships_per_doc or 5

        # Initialize graph
        self.graph = nx.DiGraph()

        # Entity and relationship storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.communities: Dict[str, Community] = {}

        # Document storage
        self.documents: Dict[str, Document] = {}

        # Verify Ollama connection
        try:
            available_models = ollama.list()
            model_names = [m.model for m in available_models.models]
            if self.model not in model_names:
                raise ValueError(f"Model {self.model} not available. Run: ollama pull {self.model}")
            if self.verbose:
                logger.info(f"GraphRAG initialized with model: {self.model}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    def _generate_entity_id(self, name: str) -> str:
        """Generate unique ID for entity"""
        return hashlib.sha256(name.lower().strip().encode()).hexdigest()[:12]

    def _extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities from a document using LLM"""
        prompt = f"""Extract key entities (concepts, technologies, organizations, people) from the following text.
For each entity, provide:
1. Name
2. Type (CONCEPT, TECHNOLOGY, PERSON, ORGANIZATION, or OTHER)
3. Brief description based on the text

Text:
{document.page_content[:2000]}

Format your response as a list, one entity per line:
ENTITY: [name] | TYPE: [type] | DESCRIPTION: [brief description]

Extract 3-{self.max_entities_per_doc} key entities:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.2, 'num_predict': 500}
        )

        entities = []
        doc_source = document.metadata.get('source', 'unknown')

        for line in response['response'].split('\n'):
            if 'ENTITY:' in line and 'TYPE:' in line:
                try:
                    parts = line.split('|')
                    name = parts[0].replace('ENTITY:', '').strip()
                    entity_type = parts[1].replace('TYPE:', '').strip() if len(parts) > 1 else "CONCEPT"
                    description = parts[2].replace('DESCRIPTION:', '').strip() if len(parts) > 2 else ""

                    if name and len(entities) < self.max_entities_per_doc:
                        entity = Entity(
                            id=self._generate_entity_id(name),
                            name=name,
                            entity_type=entity_type,
                            description=description,
                            source_docs=[doc_source]
                        )
                        entities.append(entity)
                except Exception:
                    continue

        return entities

    def _extract_relationships(self, document: Document, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        if len(entities) < 2:
            return []

        entity_names = [e.name for e in entities]

        prompt = f"""Given these entities from a document: {', '.join(entity_names)}

And this text:
{document.page_content[:1500]}

Identify relationships between the entities. For each relationship:
1. Source entity
2. Relationship type (IS_A, RELATES_TO, USES, PART_OF, ENABLES, IMPROVES, PRECEDES)
3. Target entity
4. Brief description

Format:
RELATION: [source] -> [type] -> [target] | [description]

Identify 2-{self.max_relationships_per_doc} key relationships:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.2, 'num_predict': 400}
        )

        relationships = []
        # Build entity map with normalized keys and also track original names for fuzzy matching
        entity_map = {}
        for e in entities:
            # Add lowercased, stripped version
            entity_map[e.name.lower().strip()] = e
            # Also add version with extra whitespace normalized
            entity_map[' '.join(e.name.lower().split())] = e

        doc_source = document.metadata.get('source', 'unknown')

        def find_entity(name: str) -> Optional[Entity]:
            """Find entity with fuzzy matching"""
            normalized = ' '.join(name.lower().strip().split())
            if normalized in entity_map:
                return entity_map[normalized]
            # Try substring matching for partial names
            for key, entity in entity_map.items():
                if normalized in key or key in normalized:
                    return entity
            return None

        for line in response['response'].split('\n'):
            if 'RELATION:' in line and '->' in line:
                try:
                    relation_part = line.split('RELATION:')[1].strip()
                    parts = relation_part.split('|')
                    relation_str = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ""

                    # Parse relation: source -> type -> target
                    rel_parts = relation_str.split('->')
                    if len(rel_parts) >= 3:
                        source_name = rel_parts[0].strip()
                        rel_type = rel_parts[1].strip().upper()
                        target_name = rel_parts[2].strip()

                        source_entity = find_entity(source_name)
                        target_entity = find_entity(target_name)

                        if source_entity and target_entity and len(relationships) < self.max_relationships_per_doc:
                            relationship = Relationship(
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=rel_type,
                                description=description,
                                source_doc=doc_source
                            )
                            relationships.append(relationship)
                except Exception:
                    continue

        return relationships

    def _detect_communities(self) -> None:
        """Detect communities using Louvain algorithm (simplified version)"""
        if len(self.graph.nodes()) < 2:
            return

        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        try:
            # Use greedy modularity communities as a simpler alternative to Louvain
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(undirected))
        except Exception:
            # Fallback: treat connected components as communities
            communities = list(nx.connected_components(undirected))

        self.communities = {}
        for i, community_nodes in enumerate(communities):
            community_id = f"community_{i}"

            # Find central entity (highest degree)
            central_entity = None
            max_degree = -1
            for node in community_nodes:
                degree = self.graph.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    central_entity = node

            self.communities[community_id] = Community(
                id=community_id,
                entities=list(community_nodes),
                central_entity=central_entity,
                level=0
            )

        if self.verbose:
            logger.info(f"Detected {len(self.communities)} communities")

    def _summarize_community(self, community: Community) -> str:
        """Generate a summary for a community"""
        entity_descriptions = []
        for entity_id in community.entities[:5]:  # Limit to 5 entities
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entity_descriptions.append(f"- {entity.name}: {entity.description}")

        if not entity_descriptions:
            return "No entities in community"

        prompt = f"""Summarize the following group of related concepts in 1-2 sentences:

{chr(10).join(entity_descriptions)}

Summary:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 100}
        )

        return response['response'].strip()

    def build_graph_from_documents(self, documents: List[Document]) -> Dict[str, int]:
        """
        Build knowledge graph from documents

        Args:
            documents: List of documents to process

        Returns:
            Statistics about the graph
        """
        start_time = time.time()

        if not documents:
            if self.verbose:
                logger.warning("No documents provided to build_graph_from_documents()")
            return {"documents_processed": 0, "entities": 0, "relationships": 0, "communities": 0, "build_time": 0}

        if self.verbose:
            logger.info(f"BUILDING KNOWLEDGE GRAPH from {len(documents)} documents")

        total_entities = 0
        total_relationships = 0

        for i, doc in enumerate(documents):
            if self.verbose:
                logger.info(f"Processing document {i+1}/{len(documents)}...")

            doc_id = doc.metadata.get('source', f'doc_{i}')
            self.documents[doc_id] = doc

            # Extract entities
            entities = self._extract_entities(doc)
            if self.verbose:
                logger.info(f"  Extracted {len(entities)} entities")

            # Add entities to graph and storage
            for entity in entities:
                if entity.id in self.entities:
                    # Update existing entity
                    self.entities[entity.id].source_docs.append(doc_id)
                else:
                    self.entities[entity.id] = entity
                    self.graph.add_node(
                        entity.id,
                        name=entity.name,
                        type=entity.entity_type,
                        description=entity.description
                    )
                    total_entities += 1

            # Extract relationships
            relationships = self._extract_relationships(doc, entities)
            if self.verbose:
                logger.info(f"  Extracted {len(relationships)} relationships")

            for rel in relationships:
                self.relationships.append(rel)
                self.graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight
                )
                total_relationships += 1

        # Detect communities
        if self.verbose:
            logger.info("Detecting communities...")
        self._detect_communities()

        # Summarize communities
        if self.verbose:
            logger.info("Summarizing communities...")
        for community in self.communities.values():
            community.summary = self._summarize_community(community)

        build_time = time.time() - start_time

        stats = {
            "documents_processed": len(documents),
            "entities": total_entities,
            "relationships": total_relationships,
            "communities": len(self.communities),
            "build_time": build_time
        }

        if self.verbose:
            logger.info(f"GRAPH BUILD COMPLETE: {total_entities} entities, {total_relationships} relationships, {len(self.communities)} communities in {build_time:.1f}s")

        return stats

    def _find_relevant_entities(self, query: str) -> List[str]:
        """Find entities relevant to the query"""
        if not self.entities:
            return []

        # Create entity summary for LLM
        entity_list = "\n".join([
            f"- {e.name} ({e.entity_type}): {e.description[:100]}"
            for e in list(self.entities.values())[:20]
        ])

        prompt = f"""Given this query: "{query}"

And these entities in our knowledge graph:
{entity_list}

Which entities (list their names) are most relevant to answering this query?
List 1-5 relevant entity names, one per line:"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.1, 'num_predict': 200}
        )

        # Match entity names from response using normalized comparison
        relevant_ids = []
        response_lower = response['response'].lower()

        for entity in self.entities.values():
            # Check both exact match and partial match for entity names
            entity_name_lower = entity.name.lower()
            # Handle multi-word entity names
            if entity_name_lower in response_lower or ' '.join(entity_name_lower.split()) in response_lower:
                relevant_ids.append(entity.id)

        return relevant_ids[:5]  # Limit to top 5

    def _traverse_graph(self, start_entities: List[str], max_hops: int) -> Tuple[Set[str], List[Dict]]:
        """Traverse graph from starting entities"""
        visited_entities = set(start_entities)
        visited_relationships = []
        reasoning_path = []

        current_frontier = set(start_entities)

        for hop in range(max_hops):
            if not current_frontier:
                break

            next_frontier = set()

            for entity_id in current_frontier:
                if entity_id not in self.graph:
                    continue

                # Get outgoing edges
                for successor in self.graph.successors(entity_id):
                    if successor not in visited_entities:
                        edge_data = self.graph.get_edge_data(entity_id, successor)
                        visited_entities.add(successor)
                        next_frontier.add(successor)

                        rel_info = {
                            "hop": hop + 1,
                            "from": self.entities[entity_id].name if entity_id in self.entities else entity_id,
                            "to": self.entities[successor].name if successor in self.entities else successor,
                            "relation": edge_data.get('relation_type', 'RELATES_TO'),
                            "description": edge_data.get('description', '')
                        }
                        visited_relationships.append(rel_info)
                        reasoning_path.append(rel_info)

                # Get incoming edges
                for predecessor in self.graph.predecessors(entity_id):
                    if predecessor not in visited_entities:
                        edge_data = self.graph.get_edge_data(predecessor, entity_id)
                        visited_entities.add(predecessor)
                        next_frontier.add(predecessor)

                        rel_info = {
                            "hop": hop + 1,
                            "from": self.entities[predecessor].name if predecessor in self.entities else predecessor,
                            "to": self.entities[entity_id].name if entity_id in self.entities else entity_id,
                            "relation": edge_data.get('relation_type', 'RELATES_TO'),
                            "description": edge_data.get('description', '')
                        }
                        visited_relationships.append(rel_info)
                        reasoning_path.append(rel_info)

            current_frontier = next_frontier

        return visited_entities, reasoning_path

    def query(self, query: str, retrieved_docs: Optional[List[Document]] = None) -> GraphRAGResult:
        """
        Query the knowledge graph

        Args:
            query: User query
            retrieved_docs: Optional list of retrieved documents to include as context

        Returns:
            GraphRAGResult with answer and reasoning path
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"GRAPHRAG QUERY: {query[:60]}...")

        # Find relevant starting entities
        if self.verbose:
            logger.info("Finding relevant entities...")
        start_entities = self._find_relevant_entities(query)

        if not start_entities:
            # Fallback: use most connected entities
            if self.graph.nodes():
                degrees = dict(self.graph.degree())
                start_entities = sorted(degrees, key=degrees.get, reverse=True)[:3]

        if self.verbose:
            entity_names = [self.entities[eid].name for eid in start_entities if eid in self.entities]
            logger.info(f"Starting entities: {entity_names}")

        # Traverse graph
        if self.verbose:
            logger.info(f"Traversing graph (max {self.max_hops} hops)...")
        visited_entities, reasoning_path = self._traverse_graph(start_entities, self.max_hops)

        if self.verbose:
            logger.info(f"Visited {len(visited_entities)} entities, found {len(reasoning_path)} relationships")

        # Find relevant communities
        relevant_communities = []
        for comm_id, community in self.communities.items():
            if any(eid in community.entities for eid in visited_entities):
                relevant_communities.append(comm_id)

        # Build context from visited entities and relationships
        entity_context = []
        for eid in visited_entities:
            if eid in self.entities:
                entity = self.entities[eid]
                entity_context.append(f"- {entity.name} ({entity.entity_type}): {entity.description}")

        relationship_context = []
        for rel in reasoning_path:
            relationship_context.append(f"- {rel['from']} --[{rel['relation']}]--> {rel['to']}: {rel['description']}")

        community_context = []
        for comm_id in relevant_communities[:3]:
            if comm_id in self.communities:
                community = self.communities[comm_id]
                community_context.append(f"- {comm_id}: {community.summary}")

        # Include retrieved documents if provided
        doc_context = ""
        if retrieved_docs:
            doc_snippets = []
            for i, doc in enumerate(retrieved_docs[:5]):
                content = doc.page_content[:800] if len(doc.page_content) > 800 else doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                doc_snippets.append(f"Document {i+1} ({source}):\n{content}")
            doc_context = f"\n\nRetrieved Document Content:\n{chr(10).join(doc_snippets)}"

        # Generate answer
        prompt = f"""Answer this question using the knowledge graph information and document content below.

Question: {query}
{doc_context}

Knowledge Graph - Relevant Entities:
{chr(10).join(entity_context[:10])}

Knowledge Graph - Relationships (reasoning path):
{chr(10).join(relationship_context[:10])}

{f"Community Summaries:{chr(10)}{chr(10).join(community_context)}" if community_context else ""}

Based on the document content and knowledge graph information, provide a comprehensive answer. Cite specific information from the documents when available:"""

        if self.verbose:
            logger.info("Generating answer...")

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        )

        answer = response['response'].strip()
        total_time = time.time() - start_time

        result = GraphRAGResult(
            query=query,
            answer=answer,
            reasoning_path=reasoning_path,
            entities_used=[self.entities[eid].name for eid in visited_entities if eid in self.entities],
            relationships_used=[f"{r['from']} -> {r['to']}" for r in reasoning_path],
            communities_consulted=relevant_communities,
            total_time=total_time,
            num_hops=max([r['hop'] for r in reasoning_path]) if reasoning_path else 0
        )

        if self.verbose:
            logger.info(f"GRAPHRAG RESULT: {len(result.entities_used)} entities, {result.num_hops} hops, {total_time:.1f}s")

        return result

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "communities": len(self.communities),
            "documents": len(self.documents),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }

    def clear_graph(self) -> None:
        """Clear the knowledge graph"""
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        self.communities.clear()
        self.documents.clear()
        if self.verbose:
            logger.info("Knowledge graph cleared")

    def save_graph(self, file_path: str) -> Dict[str, Any]:
        """
        Save the knowledge graph to a JSON file

        Args:
            file_path: Path to save the graph (should end in .json)

        Returns:
            Dictionary with save statistics
        """
        from pathlib import Path

        start_time = time.time()

        # Serialize entities
        entities_data = {}
        for entity_id, entity in self.entities.items():
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
        for rel in self.relationships:
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
        for comm_id, community in self.communities.items():
            communities_data[comm_id] = {
                "id": community.id,
                "entities": community.entities,
                "summary": community.summary,
                "central_entity": community.central_entity,
                "level": community.level
            }

        # Serialize documents (metadata only, content is often too large)
        documents_data = {}
        for doc_id, doc in self.documents.items():
            documents_data[doc_id] = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }

        # Build graph edges from NetworkX graph
        graph_edges = []
        for source, target, data in self.graph.edges(data=True):
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
            "stats": self.get_graph_stats()
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

        if self.verbose:
            logger.info(f"Graph saved to {file_path}: {stats['entities_saved']} entities, {stats['relationships_saved']} relationships in {save_time:.2f}s")

        return stats

    def load_graph(self, file_path: str) -> Dict[str, Any]:
        """
        Load the knowledge graph from a JSON file

        Args:
            file_path: Path to the saved graph JSON file

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
        self.clear_graph()

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
            self.entities[entity_id] = entity
            self.graph.add_node(
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
            self.relationships.append(relationship)

        # Restore graph edges
        for edge_data in graph_data.get("graph_edges", []):
            self.graph.add_edge(
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
            self.communities[comm_id] = community

        # Restore documents
        for doc_id, doc_data in graph_data.get("documents", {}).items():
            doc = Document(
                page_content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            self.documents[doc_id] = doc

        load_time = time.time() - start_time

        stats = {
            "file_path": file_path,
            "entities_loaded": len(self.entities),
            "relationships_loaded": len(self.relationships),
            "communities_loaded": len(self.communities),
            "documents_loaded": len(self.documents),
            "load_time": load_time
        }

        if self.verbose:
            logger.info(f"Graph loaded from {file_path}: {stats['entities_loaded']} entities, {stats['relationships_loaded']} relationships in {load_time:.2f}s")

        return stats


def test_graph_rag():
    """Test GraphRAG functionality"""
    print("=" * 70)
    print("GRAPHRAG TEST")
    print("=" * 70)

    try:
        # Initialize
        print("\nInitializing GraphRAG...")
        graph_rag = OllamaGraphRAG(verbose=True)
        print("SUCCESS: GraphRAG initialized")

        # Create test documents
        documents = [
            Document(
                page_content="""Machine learning is a subset of artificial intelligence (AI) that enables
                computers to learn from data without being explicitly programmed. It uses algorithms
                to identify patterns and make predictions. Deep learning is a specialized form of
                machine learning that uses neural networks with multiple layers.""",
                metadata={"source": "ml_basics"}
            ),
            Document(
                page_content="""Neural networks are computing systems inspired by biological neural networks.
                They consist of interconnected nodes (neurons) that process information. Deep learning
                uses neural networks with many layers (deep neural networks) for complex pattern recognition.
                Transformers are a type of neural network architecture that uses attention mechanisms.""",
                metadata={"source": "neural_networks"}
            ),
            Document(
                page_content="""RAG (Retrieval-Augmented Generation) is a technique that combines information
                retrieval with text generation. It retrieves relevant documents from a knowledge base
                and uses them to generate more accurate responses. GraphRAG extends this by using
                knowledge graphs for multi-hop reasoning.""",
                metadata={"source": "rag_overview"}
            )
        ]

        # Build graph
        print("\nBuilding knowledge graph...")
        stats = graph_rag.build_graph_from_documents(documents)
        print(f"Graph stats: {stats}")

        # Test query
        query = "How does deep learning relate to neural networks and machine learning?"

        print(f"\nQuery: {query}")
        result = graph_rag.query(query)

        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Answer: {result.answer[:400]}...")
        print(f"\nReasoning path:")
        for step in result.reasoning_path[:5]:
            print(f"  {step['from']} --[{step['relation']}]--> {step['to']}")
        print(f"\nEntities used: {result.entities_used}")
        print(f"Hops: {result.num_hops}")
        print(f"Time: {result.total_time:.1f}s")

        print("\n" + "=" * 70)
        print("TEST PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_graph_rag()
