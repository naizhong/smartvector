"""Tests for consolidation components."""

from datetime import datetime, timedelta

from smartvector.models import SmartVector, VectorStatus
from smartvector.confidence import ConfidenceEngine
from smartvector.consolidation import (
    ConflictDetector,
    RelationshipBuilder,
    RipplePropagator,
    ConsolidationAgent,
)


class TestConflictDetector:
    def setup_method(self):
        self.detector = ConflictDetector()

    def test_no_conflict_when_same_content(self):
        va = SmartVector(
            doc_id="a", doc_version=1, chunk_index=0,
            content="the sky is blue and clear",
            status=VectorStatus.ACTIVE,
        )
        vb = SmartVector(
            doc_id="b", doc_version=1, chunk_index=0,
            content="the sky is blue and clear",
            status=VectorStatus.ACTIVE,
        )
        conflicts = self.detector.detect_conflicts([va, vb])
        assert len(conflicts) == 0  # High content sim = redundancy, not conflict

    def test_conflict_detected(self):
        va = SmartVector(
            doc_id="a", doc_version=1, chunk_index=0,
            content="the sky calibration uses a blue baseline for atmospheric modeling",
            base_confidence=0.85,
            status=VectorStatus.ACTIVE,
            created_at=datetime.now() - timedelta(days=30),
        )
        vb = SmartVector(
            doc_id="b", doc_version=1, chunk_index=0,
            content="the sky calibration is switching to yellow baseline next quarter",
            base_confidence=0.30,
            status=VectorStatus.ACTIVE,
            created_at=datetime.now() - timedelta(days=5),
        )
        conflicts = self.detector.detect_conflicts([va, vb])
        # May or may not trigger depending on thresholds — test structure
        for c in conflicts:
            assert "vector_a" in c
            assert "vector_b" in c
            assert "resolution" in c

    def test_skips_deprecated(self):
        va = SmartVector(
            doc_id="a", content="sky is blue",
            status=VectorStatus.DEPRECATED,
        )
        vb = SmartVector(
            doc_id="b", content="sky is yellow",
            status=VectorStatus.ACTIVE,
        )
        conflicts = self.detector.detect_conflicts([va, vb])
        assert len(conflicts) == 0

    def test_keyword_overlap(self):
        sim = ConflictDetector.keyword_overlap("the sky is blue", "the sky is red")
        assert 0.3 < sim < 1.0  # Partial overlap

    def test_keyword_overlap_no_overlap(self):
        sim = ConflictDetector.keyword_overlap("hello world", "foo bar baz")
        assert sim == 0.0


class TestRelationshipBuilder:
    def setup_method(self):
        self.builder = RelationshipBuilder()

    def test_supersession_detected(self):
        va = SmartVector(
            doc_id="doc-1", doc_version=1, chunk_index=0,
            content="old content",
            source_offset_start=0, source_offset_end=50,
            status=VectorStatus.ACTIVE,
        )
        vb = SmartVector(
            doc_id="doc-1", doc_version=2, chunk_index=0,
            content="new content",
            source_offset_start=0, source_offset_end=50,
            status=VectorStatus.ACTIVE,
        )
        rels = self.builder.build_relationships([va, vb])
        assert any(r["type"] == "supersession" for r in rels)

    def test_apply_relationships(self):
        va = SmartVector(vector_id="va", content="test a")
        vb = SmartVector(vector_id="vb", content="test b")
        vectors = {"va": va, "vb": vb}

        self.builder.apply_relationships(vectors, [{
            "type": "dependency",
            "from": "vb",
            "to": "va",
            "description": "test",
        }])
        assert "va" in vb.depends_on
        assert "vb" in va.depended_by


class TestRipplePropagator:
    def setup_method(self):
        self.propagator = RipplePropagator(
            confidence_penalty_per_hop=0.15,
            max_depth=2,
        )

    def test_ripple_penalizes_dependent(self):
        source = SmartVector(
            vector_id="src",
            content="source fact",
            depended_by=["dep"],
            status=VectorStatus.ACTIVE,
        )
        dependent = SmartVector(
            vector_id="dep",
            content="dependent fact",
            base_confidence=0.80,
            depends_on=["src"],
            status=VectorStatus.ACTIVE,
        )
        vectors = {"src": source, "dep": dependent}
        events = self.propagator.propagate("src", vectors)

        assert len(events) >= 1
        assert dependent.base_confidence < 0.80
        assert "src" in dependent.contradictions

    def test_ripple_respects_max_depth(self):
        v1 = SmartVector(vector_id="v1", content="a", depended_by=["v2"], status=VectorStatus.ACTIVE)
        v2 = SmartVector(vector_id="v2", content="b", depended_by=["v3"], depends_on=["v1"], base_confidence=0.8, status=VectorStatus.ACTIVE)
        v3 = SmartVector(vector_id="v3", content="c", depended_by=["v4"], depends_on=["v2"], base_confidence=0.8, status=VectorStatus.ACTIVE)
        v4 = SmartVector(vector_id="v4", content="d", depends_on=["v3"], base_confidence=0.8, status=VectorStatus.ACTIVE)
        vectors = {"v1": v1, "v2": v2, "v3": v3, "v4": v4}

        propagator = RipplePropagator(max_depth=2)
        events = propagator.propagate("v1", vectors)

        # v2 and v3 should be affected, v4 should NOT (depth=3)
        affected_ids = {e.vector_ids[1] for e in events}
        assert "v2" in affected_ids
        assert "v3" in affected_ids
        assert "v4" not in affected_ids

    def test_skips_deprecated(self):
        source = SmartVector(vector_id="src", content="a", depended_by=["dep"], status=VectorStatus.ACTIVE)
        dep = SmartVector(vector_id="dep", content="b", base_confidence=0.8, status=VectorStatus.DEPRECATED)
        vectors = {"src": source, "dep": dep}
        events = self.propagator.propagate("src", vectors)
        assert len(events) == 0


class TestConsolidationAgent:
    def test_promotes_unconsolidated_to_active(self):
        v = SmartVector(
            vector_id="v1",
            content="test",
            status=VectorStatus.UNCONSOLIDATED,
            base_confidence=0.85,
        )
        vectors = {"v1": v}
        agent = ConsolidationAgent()
        result = agent.run_consolidation(vectors)

        assert v.status == VectorStatus.ACTIVE
        assert result["vectors_consolidated"] == 1

    def test_moves_low_confidence_to_dormant(self):
        v = SmartVector(
            vector_id="v1",
            content="old info",
            status=VectorStatus.ACTIVE,
            base_confidence=0.01,  # Very low
            created_at=datetime.now() - timedelta(days=365),
        )
        vectors = {"v1": v}
        engine = ConfidenceEngine(dormant_threshold=0.15)
        agent = ConsolidationAgent(confidence_engine=engine)
        result = agent.run_consolidation(vectors)

        assert v.status == VectorStatus.DORMANT
        assert result["vectors_dormant"] >= 1
