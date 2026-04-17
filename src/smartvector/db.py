"""
SmartVectorDB — the intelligent retrieval engine.

Combines four retrieval signals into a single ranking score:

.. code-block:: text

    final = w_sim * similarity
          + w_temp * temporal_score
          + w_conf * confidence
          + w_rel  * relational_bonus

Only returns ACTIVE vectors by default.  Deprecated and archived vectors
are excluded; dormant vectors can be included on request.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from smartvector.models import SmartVector, VectorStatus
from smartvector.confidence import ConfidenceEngine
from smartvector.factory import SmartVectorFactory
from smartvector.consolidation import ConsolidationAgent, RipplePropagator


# =============================================================================
# Retrieval result
# =============================================================================

@dataclass
class RetrievalResult:
    """A retrieval result with full score breakdown."""

    vector: SmartVector
    similarity_score: float
    temporal_score: float
    confidence_score: float
    relational_bonus: float
    final_score: float
    conflict_warnings: list[str]
    context_metadata: dict


# =============================================================================
# Database
# =============================================================================

class SmartVectorDB:
    """The complete Smart Vector Database.

    Parameters
    ----------
    similarity_weight : float
        Weight for semantic similarity signal (default 0.35).
    temporal_weight : float
        Weight for temporal recency signal (default 0.25).
    confidence_weight : float
        Weight for confidence signal (default 0.25).
    relational_weight : float
        Weight for relational-graph bonus (default 0.15).
    confidence_engine : ConfidenceEngine, optional
        Custom engine; one is created with defaults if omitted.

    Example
    -------
    >>> from smartvector import SmartVectorDB
    >>> db = SmartVectorDB()
    >>> db.ingest_document(
    ...     doc_id="spec-1", version=1,
    ...     text="The API rate limit is 1000 req/s.",
    ...     source_name="api_docs", source_type="technical_doc",
    ...     author="alice", authority=0.85,
    ... )
    >>> results = db.query("What is the rate limit?")
    >>> results[0].vector.content
    'The API rate limit is 1000 req/s.'
    """

    def __init__(
        self,
        similarity_weight: float = 0.35,
        temporal_weight: float = 0.25,
        confidence_weight: float = 0.25,
        relational_weight: float = 0.15,
        confidence_engine: ConfidenceEngine | None = None,
    ) -> None:
        self.vectors: dict[str, SmartVector] = {}
        self.recent_changes: list[str] = []

        self.w_sim = similarity_weight
        self.w_temp = temporal_weight
        self.w_conf = confidence_weight
        self.w_rel = relational_weight

        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.consolidation_agent = ConsolidationAgent(self.confidence_engine)
        self.ripple_propagator = RipplePropagator()

        self._ingestion_log: list[dict] = []

    # =========================================================================
    # Stage 1: Encoding (Hippocampal Fast-Write)
    # =========================================================================

    def ingest_document(
        self,
        doc_id: str,
        version: int,
        text: str,
        source_name: str,
        source_type: str,
        author: str,
        authority: float | None = None,
        created_at: datetime | None = None,
    ) -> list[dict]:
        """Fast-ingest a document.

        Vectors enter as ``UNCONSOLIDATED`` — available for retrieval
        immediately but not yet cross-referenced.
        """
        vectors = SmartVectorFactory.from_text(
            doc_id, version, text, source_name, source_type, author,
            authority, created_at,
        )

        results: list[dict] = []
        for vec in vectors:
            existing = self._find_existing_chunk(vec.doc_id, vec.chunk_index)
            if existing and existing.doc_version < vec.doc_version:
                existing.status = VectorStatus.DEPRECATED
                existing.superseded_by = vec.vector_id
                vec.supersedes = existing.vector_id
                vec.depends_on = existing.depends_on.copy()
                vec.depended_by = existing.depended_by.copy()

                for dep_id in existing.depended_by:
                    if dep_id in self.vectors:
                        dep_vec = self.vectors[dep_id]
                        if existing.vector_id in dep_vec.depends_on:
                            dep_vec.depends_on.remove(existing.vector_id)
                            dep_vec.depends_on.append(vec.vector_id)

                results.append({
                    "action": "superseded",
                    "old_vector": existing.vector_id,
                    "new_vector": vec.vector_id,
                })
                self.recent_changes.append(vec.vector_id)
            else:
                results.append({"action": "ingested", "vector_id": vec.vector_id})

            self.vectors[vec.vector_id] = vec

        self._ingestion_log.append({
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "version": version,
            "chunks_created": len(vectors),
        })
        return results

    def ingest_update(
        self,
        doc_id: str,
        old_text: str,
        new_text: str,
        source_name: str,
        source_type: str,
        author: str,
        authority: float | None = None,
    ) -> dict:
        """Surgical update: diff old vs new, update only affected chunks.

        This is the *Scenario A* solution — chunk-level version tracking
        via source-offset mapping.
        """
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        changed_ranges: list[dict] = []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != "equal":
                changed_ranges.append({
                    "op": op,
                    "old_range": (i1, i2),
                    "new_range": (j1, j2),
                })

        affected: set[str] = set()
        for change in changed_ranges:
            old_start, old_end = change["old_range"]
            for vec in self.vectors.values():
                if (vec.doc_id == doc_id
                        and vec.status == VectorStatus.ACTIVE
                        and vec.source_offset_start < old_end
                        and vec.source_offset_end > old_start):
                    affected.add(vec.vector_id)

        max_version = max(
            (v.doc_version for v in self.vectors.values() if v.doc_id == doc_id),
            default=0,
        )
        new_version = max_version + 1

        updated_vectors: list[SmartVector] = []
        for old_id in affected:
            old_vec = self.vectors[old_id]
            offset_delta = len(new_text) - len(old_text)
            new_content = new_text[
                old_vec.source_offset_start:
                old_vec.source_offset_end + offset_delta
            ].strip()

            new_vec = SmartVector(
                doc_id=doc_id,
                doc_version=new_version,
                chunk_index=old_vec.chunk_index,
                content=new_content,
                source_name=source_name,
                source_type=source_type,
                author=author,
                base_confidence=authority or old_vec.base_confidence,
                created_at=datetime.now(),
                source_offset_start=old_vec.source_offset_start,
                source_offset_end=old_vec.source_offset_start + len(new_content),
                half_life_days=old_vec.half_life_days,
                depends_on=old_vec.depends_on.copy(),
                depended_by=old_vec.depended_by.copy(),
                supersedes=old_id,
            )
            old_vec.status = VectorStatus.DEPRECATED
            old_vec.superseded_by = new_vec.vector_id
            self.vectors[new_vec.vector_id] = new_vec
            updated_vectors.append(new_vec)
            self.recent_changes.append(new_vec.vector_id)

        return {
            "changes_detected": len(changed_ranges),
            "chunks_affected": len(affected),
            "chunks_updated": len(updated_vectors),
            "total_chunks": sum(
                1 for v in self.vectors.values() if v.doc_id == doc_id
            ),
            "new_version": new_version,
            "updated_vector_ids": [v.vector_id for v in updated_vectors],
        }

    # =========================================================================
    # Stage 2: Consolidation
    # =========================================================================

    def run_consolidation(self) -> dict:
        """Run the background consolidation process."""
        result = self.consolidation_agent.run_consolidation(
            self.vectors, self.recent_changes,
        )
        self.recent_changes = []
        return result

    # =========================================================================
    # Stage 3: Retrieval & Reinforcement
    # =========================================================================

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_dormant: bool = False,
        reference_time: datetime | None = None,
    ) -> list[RetrievalResult]:
        """Smart retrieval with 4-signal scoring.

        Parameters
        ----------
        query_text : str
            The user's query.
        top_k : int
            Maximum results to return.
        include_dormant : bool
            Whether to include dormant vectors.
        reference_time : datetime, optional
            Override "now" for temporal scoring (useful for testing).
        """
        if reference_time is None:
            reference_time = datetime.now()

        candidates: list[RetrievalResult] = []

        for vec in self.vectors.values():
            if vec.status == VectorStatus.DEPRECATED:
                continue
            if vec.status == VectorStatus.ARCHIVED:
                continue
            if vec.status == VectorStatus.DORMANT and not include_dormant:
                continue

            sim = self._keyword_similarity(query_text, vec.content)
            temp = self._temporal_score(vec, reference_time)
            conf = self.confidence_engine.compute_current_confidence(vec, reference_time)
            rel = self._relational_bonus(vec)

            final = (
                self.w_sim * sim
                + self.w_temp * temp
                + self.w_conf * conf
                + self.w_rel * rel
            )

            warnings: list[str] = []
            for contra_id in vec.contradictions:
                if contra_id in self.vectors:
                    contra = self.vectors[contra_id]
                    warnings.append(
                        f"Conflicts with V:{contra_id} "
                        f'("{contra.content[:40]}...")'
                    )

            candidates.append(RetrievalResult(
                vector=vec,
                similarity_score=round(sim, 4),
                temporal_score=round(temp, 4),
                confidence_score=round(conf, 4),
                relational_bonus=round(rel, 4),
                final_score=round(final, 4),
                conflict_warnings=warnings,
                context_metadata={
                    "source": vec.source_name,
                    "author": vec.author,
                    "version": f"v{vec.doc_version}",
                    "age": f"{vec.age_days:.1f} days",
                    "status": vec.status.value,
                    "edges": vec.edge_count,
                    "accesses": vec.access_count,
                },
            ))

        candidates.sort(key=lambda r: r.final_score, reverse=True)
        return candidates[:top_k]

    def record_feedback(self, vector_id: str, accepted: bool) -> None:
        """Record user feedback (reconsolidation step)."""
        if vector_id in self.vectors:
            self.vectors[vector_id].record_access(accepted)

    # =========================================================================
    # LLM context builder
    # =========================================================================

    def build_llm_context(self, query_text: str, top_k: int = 3) -> str:
        """Build structured context for an LLM including all metadata.

        The output includes confidence scores, conflict warnings, and
        provenance information so the LLM can make version-aware decisions.
        """
        results = self.query(query_text, top_k=top_k)
        if not results:
            return "No relevant documents found."

        parts = [
            "=== SMART VECTOR RETRIEVAL CONTEXT ===\n"
            "INSTRUCTIONS:\n"
            "- Documents are ranked by relevance, recency, confidence, "
            "and relational importance.\n"
            "- Confidence scores reflect source authority, age decay, "
            "and user feedback.\n"
            "- If conflict warnings are present, prefer the higher-confidence "
            "source.\n"
            "- Always cite the source, version, and date in your answer.\n"
        ]

        for i, result in enumerate(results, 1):
            vec = result.vector
            warnings = ""
            if result.conflict_warnings:
                warnings = "\n  ! " + "\n  ! ".join(result.conflict_warnings)

            parts.append(
                f"\n-- Document {i} --{warnings}\n"
                f"Source: {vec.source_name} | Author: {vec.author}\n"
                f"Version: v{vec.doc_version} | Status: {vec.status.value}\n"
                f"Created: {vec.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                f"Scores: similarity={result.similarity_score}, "
                f"temporal={result.temporal_score}, "
                f"confidence={result.confidence_score}, "
                f"relational={result.relational_bonus}\n"
                f"FINAL: {result.final_score}\n"
                f"Edges: {vec.edge_count} | Accesses: {vec.access_count} | "
                f"Feedback: +{vec.positive_feedback}/-{vec.negative_feedback}\n"
                f"\n{vec.content}\n"
            )

        return "\n".join(parts)

    # =========================================================================
    # Scoring functions
    # =========================================================================

    @staticmethod
    def _keyword_similarity(query: str, content: str) -> float:
        """Keyword overlap similarity (POC).

        In production, replace with embedding cosine similarity.
        """
        q = set(query.lower().split())
        c = set(content.lower().split())
        if not q:
            return 0.0
        return len(q & c) / len(q)

    @staticmethod
    def _temporal_score(vec: SmartVector, reference_time: datetime) -> float:
        """Time-aware scoring with validity-window support."""
        age_days = (reference_time - vec.created_at).total_seconds() / 86400
        if vec.temporal_validity_end and reference_time > vec.temporal_validity_end:
            return 0.05
        return 2 ** (-age_days / 30)

    @staticmethod
    def _relational_bonus(vec: SmartVector) -> float:
        """Graph-connectivity bonus (diminishing returns, capped at 0.3)."""
        if vec.edge_count == 0:
            return 0.0
        return min(0.3, math.log(1 + vec.edge_count) * 0.1)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _find_existing_chunk(
        self,
        doc_id: str,
        chunk_index: int,
    ) -> Optional[SmartVector]:
        """Find the most recent active chunk for a doc_id + chunk_index."""
        candidates = [
            v for v in self.vectors.values()
            if v.doc_id == doc_id
            and v.chunk_index == chunk_index
            and v.status == VectorStatus.ACTIVE
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.doc_version)

    def stats(self) -> dict:
        """Return current database statistics."""
        status_counts: dict[str, int] = {}
        for vec in self.vectors.values():
            s = vec.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        active_count = max(1, status_counts.get("active", 1))
        avg_conf = sum(
            self.confidence_engine.compute_current_confidence(v)
            for v in self.vectors.values()
            if v.status == VectorStatus.ACTIVE
        ) / active_count

        return {
            "total_vectors": len(self.vectors),
            "by_status": status_counts,
            "total_edges": sum(v.edge_count for v in self.vectors.values()) // 2,
            "pending_changes": len(self.recent_changes),
            "avg_confidence": round(avg_conf, 3),
        }
