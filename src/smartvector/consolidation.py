"""
Consolidation Agent — the "sleep-time" background process.

Inspired by:
- Neuroscience: Hippocampal-neocortical memory replay during sleep
- MAGMA: Dual-stream architecture (fast ingestion + slow consolidation)
- GNN: Message passing for update propagation

The agent runs periodically and performs:
1. Conflict Detection — find contradictions between vectors
2. Relationship Building — discover depends_on / depended_by edges
3. Ripple Propagation — when a vector changes, notify dependents
4. Confidence Recalculation — update all confidence scores
5. Dormancy Management — move low-confidence vectors to dormant
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime

from smartvector.models import SmartVector, VectorStatus
from smartvector.confidence import ConfidenceEngine


# =============================================================================
# Event log
# =============================================================================

@dataclass
class ConsolidationEvent:
    """A record of an action taken by the consolidation agent."""

    timestamp: datetime
    event_type: str  # conflict, relationship, ripple, decay, merge, dormancy
    vector_ids: list[str]
    description: str
    action_taken: str


# =============================================================================
# Conflict Detector
# =============================================================================

class ConflictDetector:
    """Detects contradictory information across vectors.

    Strategy: group vectors by topic similarity, then check for pairs
    with high topic overlap but low content similarity (same subject,
    different claims).

    In production, replace the keyword heuristics with an NLI model
    or LLM-based semantic comparison.
    """

    @staticmethod
    def keyword_overlap(text_a: str, text_b: str) -> float:
        """Jaccard similarity over whitespace-tokenised words."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    @staticmethod
    def content_similarity(text_a: str, text_b: str) -> float:
        """Sequence-based similarity via :mod:`difflib`."""
        return difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def detect_conflicts(
        self,
        vectors: list[SmartVector],
        topic_threshold: float = 0.3,
        content_threshold: float = 0.4,
    ) -> list[dict]:
        """Find pairs that discuss the same topic but differ in content.

        High topic overlap + low content similarity = likely contradiction.
        High topic overlap + high content similarity = redundancy (not conflict).
        """
        active = [
            v for v in vectors
            if v.status in (VectorStatus.ACTIVE, VectorStatus.UNCONSOLIDATED)
        ]
        conflicts: list[dict] = []

        for i, va in enumerate(active):
            for vb in active[i + 1:]:
                # Skip same-doc, same-version, different-chunk (not a conflict)
                if (va.doc_id == vb.doc_id
                        and va.doc_version == vb.doc_version
                        and va.chunk_index != vb.chunk_index):
                    continue

                topic_sim = self.keyword_overlap(va.content, vb.content)
                content_sim = self.content_similarity(va.content, vb.content)

                if topic_sim > topic_threshold and content_sim < content_threshold:
                    resolution = self._resolve_conflict(va, vb)
                    conflicts.append({
                        "vector_a": va.vector_id,
                        "vector_b": vb.vector_id,
                        "topic_similarity": round(topic_sim, 3),
                        "content_similarity": round(content_sim, 3),
                        "resolution": resolution,
                    })

        return conflicts

    @staticmethod
    def _resolve_conflict(va: SmartVector, vb: SmartVector) -> dict:
        """Determine which vector wins a conflict (majority vote)."""
        reasons: list[str] = []

        recency_winner = va if va.created_at > vb.created_at else vb
        reasons.append(f"V:{recency_winner.vector_id} is more recent")

        if va.base_confidence > vb.base_confidence:
            authority_winner = va
            reasons.append(
                f"V:{va.vector_id} has higher authority "
                f"({va.base_confidence} vs {vb.base_confidence})"
            )
        else:
            authority_winner = vb
            reasons.append(
                f"V:{vb.vector_id} has higher authority "
                f"({vb.base_confidence} vs {va.base_confidence})"
            )

        feedback_winner = va if va.feedback_ratio > vb.feedback_ratio else vb

        winners = [recency_winner, authority_winner, feedback_winner]
        winner = max(set(winners), key=winners.count)

        return {
            "preferred": winner.vector_id,
            "demoted": va.vector_id if winner == vb else vb.vector_id,
            "reasons": reasons,
            "confidence": "high" if winners.count(winner) == 3 else "medium",
        }


# =============================================================================
# Relationship Builder
# =============================================================================

class RelationshipBuilder:
    """Discovers and builds graph edges between vectors.

    Relationship types:

    - **depends_on** / **depended_by**: content references
    - **supersedes**: newer version of the same chunk
    - **contradicts**: conflicting information
    """

    def build_relationships(
        self,
        vectors: list[SmartVector],
        similarity_threshold: float = 0.35,
    ) -> list[dict]:
        """Find relationships based on content similarity and lineage."""
        relationships: list[dict] = []
        active = [v for v in vectors if v.status != VectorStatus.ARCHIVED]

        for i, va in enumerate(active):
            for vb in active[i + 1:]:
                # Same doc, different versions → supersession
                if va.doc_id == vb.doc_id and va.doc_version != vb.doc_version:
                    if va.doc_version < vb.doc_version:
                        older, newer = va, vb
                    else:
                        older, newer = vb, va

                    if self._offsets_overlap(older, newer):
                        relationships.append({
                            "type": "supersession",
                            "from": newer.vector_id,
                            "to": older.vector_id,
                            "description": (
                                f"v{newer.doc_version} supersedes v{older.doc_version}"
                            ),
                        })
                        continue

                # Different docs → check content dependency
                sim = self._keyword_overlap(va.content, vb.content)
                if sim > similarity_threshold:
                    if va.base_confidence >= vb.base_confidence:
                        relationships.append({
                            "type": "dependency",
                            "from": vb.vector_id,
                            "to": va.vector_id,
                            "description": (
                                f"V:{vb.vector_id} references "
                                f"V:{va.vector_id} (sim={sim:.2f})"
                            ),
                        })
                    else:
                        relationships.append({
                            "type": "dependency",
                            "from": va.vector_id,
                            "to": vb.vector_id,
                            "description": (
                                f"V:{va.vector_id} references "
                                f"V:{vb.vector_id} (sim={sim:.2f})"
                            ),
                        })

        return relationships

    @staticmethod
    def apply_relationships(
        vectors: dict[str, SmartVector],
        relationships: list[dict],
    ) -> None:
        """Mutate vector objects to record discovered relationships."""
        for rel in relationships:
            from_id = rel["from"]
            to_id = rel["to"]
            if from_id not in vectors or to_id not in vectors:
                continue
            if rel["type"] == "supersession":
                vectors[from_id].supersedes = to_id
                vectors[to_id].superseded_by = from_id
            elif rel["type"] == "dependency":
                if to_id not in vectors[from_id].depends_on:
                    vectors[from_id].depends_on.append(to_id)
                if from_id not in vectors[to_id].depended_by:
                    vectors[to_id].depended_by.append(from_id)

    @staticmethod
    def _offsets_overlap(a: SmartVector, b: SmartVector) -> bool:
        return (a.source_offset_start < b.source_offset_end
                and a.source_offset_end > b.source_offset_start)

    @staticmethod
    def _keyword_overlap(text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)


# =============================================================================
# Ripple Propagator
# =============================================================================

class RipplePropagator:
    """Propagate changes through the knowledge graph (GNN-style message passing).

    When a vector is updated or deprecated, dependent vectors receive a
    confidence penalty that attenuates with each hop.  This ensures that
    downstream knowledge is flagged for review when upstream facts change.

    Parameters
    ----------
    confidence_penalty_per_hop : float
        Base penalty applied at each hop (attenuated by depth).
    max_depth : int
        Maximum propagation depth to prevent cascade explosion.
    """

    def __init__(
        self,
        confidence_penalty_per_hop: float = 0.15,
        max_depth: int = 2,
    ) -> None:
        self.penalty_per_hop = confidence_penalty_per_hop
        self.max_depth = max_depth

    def propagate(
        self,
        changed_vector_id: str,
        vectors: dict[str, SmartVector],
        change_type: str = "updated",
    ) -> list[ConsolidationEvent]:
        """Propagate a change from one vector to its dependents."""
        events: list[ConsolidationEvent] = []
        visited: set[str] = set()
        self._propagate_recursive(
            changed_vector_id, vectors, change_type,
            changed_vector_id=changed_vector_id,
            depth=0, events=events, visited=visited,
        )
        return events

    def _propagate_recursive(
        self,
        vector_id: str,
        vectors: dict[str, SmartVector],
        change_type: str,
        changed_vector_id: str,
        depth: int,
        events: list[ConsolidationEvent],
        visited: set[str],
    ) -> None:
        if depth >= self.max_depth or vector_id in visited:
            return
        visited.add(vector_id)
        if vector_id not in vectors:
            return

        source_vec = vectors[vector_id]

        for dep_id in source_vec.depended_by:
            if dep_id in visited or dep_id not in vectors:
                continue

            dep_vec = vectors[dep_id]
            if dep_vec.status in (VectorStatus.DEPRECATED, VectorStatus.ARCHIVED):
                continue

            penalty = self.penalty_per_hop * (1.0 / (depth + 1))
            old_conf = dep_vec.base_confidence
            dep_vec.base_confidence = max(0.05, dep_vec.base_confidence - penalty)

            if changed_vector_id not in dep_vec.contradictions:
                dep_vec.contradictions.append(changed_vector_id)

            events.append(ConsolidationEvent(
                timestamp=datetime.now(),
                event_type="ripple",
                vector_ids=[changed_vector_id, dep_id],
                description=(
                    f"Change in V:{changed_vector_id} ({change_type}) "
                    f"rippled to V:{dep_id} at depth {depth + 1}"
                ),
                action_taken=(
                    f"Confidence reduced {old_conf:.2f} -> {dep_vec.base_confidence:.2f} "
                    f"(penalty: {penalty:.2f}). Flagged for review."
                ),
            ))

            self._propagate_recursive(
                dep_id, vectors, change_type,
                changed_vector_id=changed_vector_id,
                depth=depth + 1, events=events, visited=visited,
            )


# =============================================================================
# Consolidation Agent (orchestrator)
# =============================================================================

class ConsolidationAgent:
    """Background agent that orchestrates all consolidation tasks.

    Call :meth:`run_consolidation` periodically (e.g. every hour, or
    nightly) to keep the knowledge graph healthy.

    Pipeline:
    1. Recalculate confidence for all active vectors
    2. Detect conflicts
    3. Discover relationships
    4. Propagate ripples from recent changes
    5. Promote ``UNCONSOLIDATED`` → ``ACTIVE``
    """

    def __init__(self, confidence_engine: ConfidenceEngine | None = None) -> None:
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.conflict_detector = ConflictDetector()
        self.relationship_builder = RelationshipBuilder()
        self.ripple_propagator = RipplePropagator()
        self.event_log: list[ConsolidationEvent] = []

    def run_consolidation(
        self,
        vectors: dict[str, SmartVector],
        recent_changes: list[str] | None = None,
    ) -> dict:
        """Run a full consolidation cycle.

        Parameters
        ----------
        vectors : dict[str, SmartVector]
            The entire vector store (mutated in place).
        recent_changes : list[str], optional
            Vector IDs that changed since the last run.

        Returns
        -------
        dict
            Summary of actions taken.
        """
        summary: dict = {
            "timestamp": datetime.now().isoformat(),
            "vectors_processed": len(vectors),
            "confidence_updates": 0,
            "conflicts_found": 0,
            "relationships_built": 0,
            "ripples_propagated": 0,
            "vectors_dormant": 0,
            "vectors_consolidated": 0,
            "events": [],
        }
        recent_changes = recent_changes or []

        # Step 1: Recalculate confidence
        for vec in vectors.values():
            if vec.status in (VectorStatus.ACTIVE, VectorStatus.UNCONSOLIDATED):
                new_conf = self.confidence_engine.compute_current_confidence(vec)
                if self.confidence_engine.should_go_dormant(new_conf):
                    vec.status = VectorStatus.DORMANT
                    summary["vectors_dormant"] += 1
                    self._log_event(
                        "dormancy", [vec.vector_id],
                        f"V:{vec.vector_id} confidence={new_conf:.3f} below threshold",
                        "Moved to DORMANT status",
                    )
                summary["confidence_updates"] += 1

        # Step 2: Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(list(vectors.values()))
        for conflict in conflicts:
            va_id, vb_id = conflict["vector_a"], conflict["vector_b"]
            if va_id in vectors and vb_id in vectors:
                if vb_id not in vectors[va_id].contradictions:
                    vectors[va_id].contradictions.append(vb_id)
                if va_id not in vectors[vb_id].contradictions:
                    vectors[vb_id].contradictions.append(va_id)
            resolution = conflict["resolution"]
            self._log_event(
                "conflict", [va_id, vb_id],
                f"Conflict (topic={conflict['topic_similarity']}, "
                f"content={conflict['content_similarity']})",
                f"Preferred: V:{resolution['preferred']} "
                f"({', '.join(resolution['reasons'])})",
            )
            summary["conflicts_found"] += 1

        # Step 3: Build relationships
        relationships = self.relationship_builder.build_relationships(
            list(vectors.values())
        )
        self.relationship_builder.apply_relationships(vectors, relationships)
        for rel in relationships:
            self._log_event(
                "relationship", [rel["from"], rel["to"]],
                rel["description"], f"Type: {rel['type']}",
            )
            summary["relationships_built"] += 1

        # Step 4: Propagate ripples
        for changed_id in recent_changes:
            if changed_id in vectors:
                events = self.ripple_propagator.propagate(changed_id, vectors)
                self.event_log.extend(events)
                summary["events"].extend([{
                    "type": e.event_type,
                    "description": e.description,
                    "action": e.action_taken,
                } for e in events])
                summary["ripples_propagated"] += len(events)

        # Step 5: Promote unconsolidated → active
        for vec in vectors.values():
            if vec.status == VectorStatus.UNCONSOLIDATED:
                vec.status = VectorStatus.ACTIVE
                summary["vectors_consolidated"] += 1

        summary["events"] = summary["events"][:20]
        return summary

    def _log_event(
        self,
        event_type: str,
        vector_ids: list[str],
        description: str,
        action: str,
    ) -> None:
        self.event_log.append(ConsolidationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            vector_ids=vector_ids,
            description=description,
            action_taken=action,
        ))
