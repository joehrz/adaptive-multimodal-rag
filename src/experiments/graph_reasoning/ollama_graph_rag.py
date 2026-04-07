"""
GraphRAG Implementation with Ollama
Builds knowledge graphs from documents for multi-hop reasoning

Based on "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"

Features:
- Entity extraction from documents
- Relationship identification
- Community detection (greedy modularity)
- Multi-hop graph traversal for complex queries
"""

import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, TYPE_CHECKING

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

from src.experiments.graph_reasoning.models import (
    Entity, Relationship, Community, GraphRAGResult
)
from src.experiments.graph_reasoning.graph_builder import GraphBuilder
from src.experiments.graph_reasoning import graph_persistence

logger = logging.getLogger(__name__)


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
        graph_building_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_hops: Optional[int] = None,
        verbose: Optional[bool] = None,
        config: Optional['Config'] = None,
        timeout: Optional[int] = None,
        max_entities_per_doc: Optional[int] = None,
        max_relationships_per_doc: Optional[int] = None,
        max_documents: Optional[int] = None,
    ):
        """
        Initialize GraphRAG system

        Args:
            model: Ollama model to use
            graph_building_model: Smaller/faster model for graph building (entity/relationship extraction)
            temperature: Generation temperature
            max_tokens: Maximum tokens for generation
            max_hops: Maximum hops in graph traversal
            verbose: Enable verbose logging
            config: Optional Config object (uses global config if not provided)
            timeout: Timeout for LLM calls in seconds
            max_entities_per_doc: Maximum entities to extract per document
            max_relationships_per_doc: Maximum relationships to extract per document
            max_documents: Maximum document chunks to process for graph building
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
            self.max_documents = max_documents if max_documents is not None else config.strategies.graphrag.max_documents
            self.graph_building_model = graph_building_model or config.strategies.graphrag.graph_building_model or self.model
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
            self.max_documents = max_documents or 30
            self.graph_building_model = graph_building_model or self.model

        # Initialize graph
        self.graph = nx.DiGraph()

        # Entity and relationship storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.communities: Dict[str, Community] = {}

        # Document storage
        self.documents: Dict[str, Document] = {}

        # Initialize graph builder
        self._builder = GraphBuilder(
            graph_building_model=self.graph_building_model,
            max_entities_per_doc=self.max_entities_per_doc,
            max_relationships_per_doc=self.max_relationships_per_doc,
            verbose=self.verbose,
        )

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
        return GraphBuilder.generate_entity_id(name)

    def _extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities from a document using LLM"""
        return self._builder.extract_entities(document)

    def _extract_relationships(self, document: Document, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        return self._builder.extract_relationships(document, entities)

    def _detect_communities(self) -> None:
        """Detect communities using greedy modularity algorithm"""
        self.communities = self._builder.detect_communities(
            self.graph, self.entities, self.verbose
        )

    def _summarize_community(self, community: Community) -> str:
        """Generate a summary for a community"""
        return self._builder.summarize_community(community, self.entities)

    def build_graph_from_documents(self, documents: List[Document], max_documents: Optional[int] = None) -> Dict[str, int]:
        """
        Build knowledge graph from documents

        Args:
            documents: List of documents to process
            max_documents: Maximum chunks to process (overrides config). Samples evenly across the document list.

        Returns:
            Statistics about the graph
        """
        start_time = time.time()

        if not documents:
            if self.verbose:
                logger.warning("No documents provided to build_graph_from_documents()")
            return {"documents_processed": 0, "entities": 0, "relationships": 0, "communities": 0, "build_time": 0}

        limit = max_documents if max_documents is not None else self.max_documents
        if limit is not None and len(documents) > limit:
            step = len(documents) / limit
            documents = [documents[int(i * step)] for i in range(limit)]
            if self.verbose:
                logger.info(f"Sampled {len(documents)} chunks from original set (max_documents={limit})")

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
            options={'temperature': 0.1, 'num_predict': 200},
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

Based on the document content and knowledge graph information, provide an answer. Cite specific information from the documents when available:"""

        if self.verbose:
            logger.info("Generating answer...")

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            },
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
        return graph_persistence.save_graph(
            file_path=file_path,
            graph=self.graph,
            entities=self.entities,
            relationships=self.relationships,
            communities=self.communities,
            documents=self.documents,
            get_graph_stats=self.get_graph_stats,
            verbose=self.verbose,
        )

    def load_graph(self, file_path: str) -> Dict[str, Any]:
        """
        Load the knowledge graph from a JSON file

        Args:
            file_path: Path to the saved graph JSON file

        Returns:
            Dictionary with load statistics
        """
        return graph_persistence.load_graph(
            file_path=file_path,
            graph=self.graph,
            entities=self.entities,
            relationships=self.relationships,
            communities=self.communities,
            documents=self.documents,
            clear_graph_fn=self.clear_graph,
            verbose=self.verbose,
        )


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
