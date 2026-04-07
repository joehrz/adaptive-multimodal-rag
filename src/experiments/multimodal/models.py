"""
Data models for multimodal RAG processing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ContentType(Enum):
    """Types of content in multimodal documents"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"


@dataclass
class ExtractedContent:
    """Represents extracted content from a document"""
    content_type: ContentType
    content: str  # Text description or actual text
    metadata: Dict
    page_number: int
    source_path: Optional[str] = None
