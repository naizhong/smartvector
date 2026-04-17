"""
Core data models for SmartVector.

SmartVector is the fundamental unit — a self-aware embedding that carries
temporal, confidence, and relational metadata alongside its content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class VectorStatus(Enum):
    """Lifecycle states — mirrors neuroscience memory stages."""

    UNCONSOLIDATED = "unconsolidated"  # Just ingested, not cross-referenced
    ACTIVE = "active"                  # Consolidated and serving queries
    DORMANT = "dormant"                # Low confidence, deprioritized
    DEPRECATED = "deprecated"          # Explicitly superseded by newer version
    ARCHIVED = "archived"              # Permanently stored for audit trail


class SourceAuthority(Enum):
    """Predefined authority levels by source type.

    These serve as default ``base_confidence`` values when ingesting
    documents.  Higher authority means the system trusts the source more
    and its vectors decay more slowly relative to lower-authority sources.
    """

    OFFICIAL_DB = 0.95       # Company database, CRM, ERP
    POLICY_DOC = 0.90        # HR policies, compliance docs
    TECHNICAL_DOC = 0.85     # Technical specifications, API docs
    WIKI = 0.75              # Internal wiki, Confluence
    EMAIL = 0.50             # Email communications
    MEETING_NOTES = 0.45     # Meeting transcripts
    SLACK = 0.30             # Slack messages, chat
    UNKNOWN = 0.20           # Unclassified source


@dataclass
class SmartVector:
    """A self-aware embedding unit.

    Each SmartVector knows:

    - **WHAT** it means — semantic content and embedding
    - **WHEN** it's valid — temporal metadata with optional validity window
    - **HOW confident** it should be — authority + decay + feedback
    - **WHO** it's related to — graph edges to other vectors

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    doc_version : int
        Version number within that document's lineage.
    chunk_index : int
        Position of this chunk within the document.
    content : str
        The raw text content of this chunk.
    source_name : str
        Human-readable name of the source (e.g. ``"engineering_wiki"``).
    source_type : str
        Category of the source (e.g. ``"technical_doc"``, ``"slack"``).
    author : str
        Who created or last modified this content.
    base_confidence : float
        Starting confidence, typically from :class:`SourceAuthority`.
    """

    # ── Identity ──
    vector_id: str = field(default_factory=lambda: str(uuid4())[:12])
    doc_id: str = ""
    doc_version: int = 1
    chunk_index: int = 0

    # ── Content ──
    content: str = ""
    content_hash: str = ""
    semantic_embedding: list[float] = field(default_factory=list)

    # ── Source ──
    source_name: str = ""
    source_type: str = ""
    author: str = ""

    # ── Property 1: Temporal Awareness ──
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    temporal_validity_start: Optional[datetime] = None
    temporal_validity_end: Optional[datetime] = None

    # ── Property 2: Confidence ──
    base_confidence: float = 0.8
    access_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    last_accessed: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    half_life_days: float = 30.0

    # ── Property 3: Relational Awareness ──
    depends_on: list[str] = field(default_factory=list)
    depended_by: list[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    contradictions: list[str] = field(default_factory=list)

    # ── Lifecycle ──
    status: VectorStatus = VectorStatus.UNCONSOLIDATED
    source_offset_start: int = 0
    source_offset_end: int = 0

    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]

    # ── Convenience properties ──

    @property
    def age_days(self) -> float:
        """Days since this vector was created."""
        return (datetime.now() - self.created_at).total_seconds() / 86400

    @property
    def edge_count(self) -> int:
        """Total number of graph edges (depends_on + depended_by)."""
        return len(self.depends_on) + len(self.depended_by)

    @property
    def feedback_ratio(self) -> float:
        """Ratio of positive feedback.  0.5 if no feedback yet."""
        total = self.positive_feedback + self.negative_feedback
        if total == 0:
            return 0.5
        return self.positive_feedback / total

    def record_access(self, accepted: bool = True) -> None:
        """Record a retrieval event with user feedback.

        This is the *reconsolidation* step — accessing a memory modifies it.
        """
        self.access_count += 1
        self.last_accessed = datetime.now()
        if accepted:
            self.positive_feedback += 1
        else:
            self.negative_feedback += 1

    def validate(self) -> None:
        """Mark as human-validated (resets the effective decay clock)."""
        self.last_validated = datetime.now()

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary for inspection / logging."""
        return {
            "vector_id": self.vector_id,
            "doc_id": self.doc_id,
            "doc_version": self.doc_version,
            "chunk_index": self.chunk_index,
            "content": (self.content[:80] + "...") if len(self.content) > 80 else self.content,
            "status": self.status.value,
            "base_confidence": self.base_confidence,
            "age_days": round(self.age_days, 1),
            "access_count": self.access_count,
            "feedback": f"+{self.positive_feedback}/-{self.negative_feedback}",
            "edges": self.edge_count,
            "depends_on": self.depends_on,
            "depended_by": self.depended_by,
            "contradictions": self.contradictions,
        }
