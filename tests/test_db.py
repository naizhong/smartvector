"""Tests for SmartVectorDB — the full lifecycle."""

from datetime import datetime, timedelta

from smartvector.db import SmartVectorDB, RetrievalResult
from smartvector.models import SmartVector, VectorStatus


class TestSmartVectorDB:
    def setup_method(self):
        self.db = SmartVectorDB()

    def test_ingest_creates_vectors(self):
        results = self.db.ingest_document(
            doc_id="doc-1", version=1,
            text="The API rate limit is 1000 requests per second. This applies to all endpoints.",
            source_name="api_docs", source_type="technical_doc",
            author="alice", authority=0.85,
        )
        assert len(results) >= 1
        assert results[0]["action"] == "ingested"
        assert self.db.stats()["total_vectors"] >= 1

    def test_ingest_vectors_start_unconsolidated(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1, text="Some content here.",
            source_name="wiki", source_type="wiki", author="bob",
        )
        for v in self.db.vectors.values():
            assert v.status == VectorStatus.UNCONSOLIDATED

    def test_consolidation_promotes_to_active(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1, text="Some content here.",
            source_name="wiki", source_type="wiki", author="bob",
        )
        self.db.run_consolidation()
        for v in self.db.vectors.values():
            if v.status != VectorStatus.DORMANT:
                assert v.status == VectorStatus.ACTIVE

    def test_query_returns_results(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1,
            text="The sky color calibration uses blue baseline for modeling.",
            source_name="wiki", source_type="technical_doc",
            author="alice", authority=0.85,
        )
        self.db.run_consolidation()

        results = self.db.query("What color baseline?", top_k=3)
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].final_score > 0

    def test_query_excludes_deprecated(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1,
            text="Sky is blue for atmospheric modeling.",
            source_name="wiki", source_type="wiki", author="alice",
            authority=0.85,
        )
        self.db.run_consolidation()

        # Deprecate manually
        for v in self.db.vectors.values():
            v.status = VectorStatus.DEPRECATED

        results = self.db.query("sky blue", top_k=3)
        assert len(results) == 0

    def test_surgical_update(self):
        old_text = "Enterprise plan costs $99 per month. Volume discounts available."
        self.db.ingest_document(
            doc_id="pricing", version=1, text=old_text,
            source_name="docs", source_type="technical_doc",
            author="alice", authority=0.85,
        )
        self.db.run_consolidation()

        new_text = "Enterprise plan costs $149 per month. Volume discounts available."
        result = self.db.ingest_update(
            doc_id="pricing", old_text=old_text, new_text=new_text,
            source_name="docs", source_type="technical_doc",
            author="alice_v2", authority=0.90,
        )
        assert result["changes_detected"] >= 1
        assert result["new_version"] == 2

        # Check old vectors deprecated
        deprecated = [
            v for v in self.db.vectors.values()
            if v.doc_id == "pricing" and v.status == VectorStatus.DEPRECATED
        ]
        assert len(deprecated) >= 1

    def test_record_feedback(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1, text="Test content.",
            source_name="wiki", source_type="wiki", author="bob",
        )
        vid = list(self.db.vectors.keys())[0]

        self.db.record_feedback(vid, accepted=True)
        assert self.db.vectors[vid].positive_feedback == 1
        assert self.db.vectors[vid].access_count == 1

    def test_build_llm_context(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1,
            text="The rate limit is 1000 requests per second.",
            source_name="api_docs", source_type="technical_doc",
            author="alice", authority=0.85,
        )
        self.db.run_consolidation()

        context = self.db.build_llm_context("rate limit", top_k=2)
        assert "SMART VECTOR RETRIEVAL CONTEXT" in context
        assert "api_docs" in context

    def test_build_llm_context_empty(self):
        context = self.db.build_llm_context("anything")
        assert context == "No relevant documents found."

    def test_temporal_decay_affects_ranking(self):
        """Older vectors should score lower with temporal decay."""
        self.db.ingest_document(
            doc_id="old-report", version=1,
            text="The quarterly revenue report shows growth of 15 percent year over year.",
            source_name="finance", source_type="wiki", author="alice",
            authority=0.85,
            created_at=datetime.now() - timedelta(days=90),
        )
        self.db.ingest_document(
            doc_id="new-report", version=1,
            text="The quarterly revenue report shows growth of 22 percent year over year.",
            source_name="finance", source_type="wiki", author="bob",
            authority=0.85,
            created_at=datetime.now() - timedelta(days=1),
        )
        # Query without consolidation to avoid dormancy from conflict detection
        results = self.db.query("quarterly revenue report growth", top_k=2)
        assert len(results) == 2
        # The newer one should rank higher (same authority, better temporal)
        assert results[0].temporal_score > results[1].temporal_score

    def test_stats(self):
        self.db.ingest_document(
            doc_id="doc-1", version=1, text="Hello world.",
            source_name="wiki", source_type="wiki", author="bob",
        )
        stats = self.db.stats()
        assert stats["total_vectors"] >= 1
        assert "by_status" in stats
        assert "total_edges" in stats


class TestScoringFunctions:
    def test_keyword_similarity(self):
        sim = SmartVectorDB._keyword_similarity("sky blue color", "the sky is blue")
        assert sim > 0

    def test_keyword_similarity_no_match(self):
        sim = SmartVectorDB._keyword_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_keyword_similarity_empty_query(self):
        sim = SmartVectorDB._keyword_similarity("", "some content")
        assert sim == 0.0

    def test_relational_bonus_no_edges(self):
        v = SmartVector(content="test")
        bonus = SmartVectorDB._relational_bonus(v)
        assert bonus == 0.0

    def test_relational_bonus_with_edges(self):
        v = SmartVector(content="test", depends_on=["a", "b", "c"])
        bonus = SmartVectorDB._relational_bonus(v)
        assert 0 < bonus <= 0.3

    def test_relational_bonus_capped(self):
        v = SmartVector(content="test", depends_on=["a"] * 100)
        bonus = SmartVectorDB._relational_bonus(v)
        assert bonus <= 0.3
