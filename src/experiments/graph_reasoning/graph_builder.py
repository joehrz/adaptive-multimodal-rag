"""Graph building: entity extraction, relationship extraction, community detection"""

import hashlib
import logging
from typing import Dict, List, Optional

import networkx as nx

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from langchain.schema import Document

from src.experiments.graph_reasoning.models import Entity, Relationship, Community

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Handles entity extraction, relationship extraction, and community detection."""

    def __init__(
        self,
        graph_building_model: str,
        max_entities_per_doc: int,
        max_relationships_per_doc: int,
        verbose: bool,
    ):
        self.graph_building_model = graph_building_model
        self.max_entities_per_doc = max_entities_per_doc
        self.max_relationships_per_doc = max_relationships_per_doc
        self.verbose = verbose

    @staticmethod
    def generate_entity_id(name: str) -> str:
        """Generate unique ID for entity"""
        return hashlib.sha256(name.lower().strip().encode()).hexdigest()[:12]

    def extract_entities(self, document: Document) -> List[Entity]:
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

        try:
            response = ollama.generate(
                model=self.graph_building_model,
                prompt=prompt,
                options={'temperature': 0.2, 'num_predict': 500},
            )
        except Exception as e:
            logger.warning(f"Entity extraction LLM call failed: {e}")
            return []

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
                            id=self.generate_entity_id(name),
                            name=name,
                            entity_type=entity_type,
                            description=description,
                            source_docs=[doc_source]
                        )
                        entities.append(entity)
                except Exception:
                    continue

        return entities

    def extract_relationships(self, document: Document, entities: List[Entity]) -> List[Relationship]:
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

        try:
            response = ollama.generate(
                model=self.graph_building_model,
                prompt=prompt,
                options={'temperature': 0.2, 'num_predict': 400},
            )
        except Exception as e:
            logger.warning(f"Relationship extraction LLM call failed: {e}")
            return []

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

    def detect_communities(
        self,
        graph: nx.DiGraph,
        entities: Dict[str, Entity],
        verbose: bool,
    ) -> Dict[str, Community]:
        """Detect communities using greedy modularity algorithm"""
        if len(graph.nodes()) < 2:
            return {}

        # Convert to undirected for community detection
        undirected = graph.to_undirected()

        try:
            # Use greedy modularity communities as a simpler alternative to Louvain
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(undirected))
        except Exception:
            # Fallback: treat connected components as communities
            communities = list(nx.connected_components(undirected))

        result = {}
        for i, community_nodes in enumerate(communities):
            community_id = f"community_{i}"

            # Find central entity (highest degree)
            central_entity = None
            max_degree = -1
            for node in community_nodes:
                degree = graph.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    central_entity = node

            result[community_id] = Community(
                id=community_id,
                entities=list(community_nodes),
                central_entity=central_entity,
                level=0
            )

        if verbose:
            logger.info(f"Detected {len(result)} communities")

        return result

    def summarize_community(self, community: Community, entities: Dict[str, Entity]) -> str:
        """Generate a summary for a community"""
        entity_descriptions = []
        for entity_id in community.entities[:5]:  # Limit to 5 entities
            if entity_id in entities:
                entity = entities[entity_id]
                entity_descriptions.append(f"- {entity.name}: {entity.description}")

        if not entity_descriptions:
            return "No entities in community"

        prompt = f"""Summarize the following group of related concepts in 1-2 sentences:

{chr(10).join(entity_descriptions)}

Summary:"""

        response = ollama.generate(
            model=self.graph_building_model,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 100},
        )

        return response['response'].strip()
