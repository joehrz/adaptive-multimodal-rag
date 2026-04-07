"""Data models for GraphRAG"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


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
