"""
SmartVector — Self-aware vector embeddings for RAG systems.

Vectors that know WHAT they mean, WHEN they're valid,
HOW confident they should be, and WHO they're related to.
"""

__version__ = "0.1.0"

from smartvector.models import (
    SmartVector,
    VectorStatus,
    SourceAuthority,
)
from smartvector.confidence import ConfidenceEngine
from smartvector.consolidation import (
    ConflictDetector,
    RelationshipBuilder,
    RipplePropagator,
    ConsolidationAgent,
    ConsolidationEvent,
)
from smartvector.db import SmartVectorDB, RetrievalResult
from smartvector.factory import SmartVectorFactory

__all__ = [
    "SmartVector",
    "SmartVectorDB",
    "SmartVectorFactory",
    "VectorStatus",
    "SourceAuthority",
    "ConfidenceEngine",
    "ConflictDetector",
    "RelationshipBuilder",
    "RipplePropagator",
    "ConsolidationAgent",
    "ConsolidationEvent",
    "RetrievalResult",
]
