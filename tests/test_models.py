"""Tests for SmartVector models."""

from datetime import datetime, timedelta

from smartvector.models import SmartVector, VectorStatus, SourceAuthority


class TestVectorStatus:
    def test_all_statuses_exist(self):
        assert VectorStatus.UNCONSOLIDATED.value == "unconsolidated"
        assert VectorStatus.ACTIVE.value == "active"
        assert VectorStatus.DORMANT.value == "dormant"
        assert VectorStatus.DEPRECATED.value == "deprecated"
        assert VectorStatus.ARCHIVED.value == "archived"


class TestSourceAuthority:
    def test_hierarchy(self):
        assert SourceAuthority.OFFICIAL_DB.value > SourceAuthority.WIKI.value
        assert SourceAuthority.WIKI.value > SourceAuthority.SLACK.value
        assert SourceAuthority.SLACK.value > SourceAuthority.UNKNOWN.value


class TestSmartVector:
    def test_defaults(self):
        v = SmartVector(content="hello world")
        assert v.status == VectorStatus.UNCONSOLIDATED
        assert v.base_confidence == 0.8
        assert v.access_count == 0
        assert len(v.vector_id) == 12

    def test_content_hash_generated(self):
        v = SmartVector(content="test content")
        assert v.content_hash != ""
        assert len(v.content_hash) == 12

    def test_content_hash_deterministic(self):
        v1 = SmartVector(content="same content", content_hash="")
        v2 = SmartVector(content="same content", content_hash="")
        assert v1.content_hash == v2.content_hash

    def test_edge_count(self):
        v = SmartVector(
            depends_on=["a", "b"],
            depended_by=["c"],
        )
        assert v.edge_count == 3

    def test_feedback_ratio_neutral(self):
        v = SmartVector()
        assert v.feedback_ratio == 0.5

    def test_feedback_ratio_positive(self):
        v = SmartVector(positive_feedback=3, negative_feedback=1)
        assert v.feedback_ratio == 0.75

    def test_record_access_positive(self):
        v = SmartVector()
        v.record_access(accepted=True)
        assert v.access_count == 1
        assert v.positive_feedback == 1
        assert v.negative_feedback == 0
        assert v.last_accessed is not None

    def test_record_access_negative(self):
        v = SmartVector()
        v.record_access(accepted=False)
        assert v.access_count == 1
        assert v.positive_feedback == 0
        assert v.negative_feedback == 1

    def test_validate(self):
        v = SmartVector()
        assert v.last_validated is None
        v.validate()
        assert v.last_validated is not None

    def test_to_dict(self):
        v = SmartVector(
            doc_id="doc-1",
            doc_version=2,
            content="short",
        )
        d = v.to_dict()
        assert d["doc_id"] == "doc-1"
        assert d["doc_version"] == 2
        assert d["content"] == "short"
        assert d["status"] == "unconsolidated"

    def test_to_dict_truncates_long_content(self):
        v = SmartVector(content="x" * 200)
        d = v.to_dict()
        assert d["content"].endswith("...")
        assert len(d["content"]) == 83  # 80 + "..."

    def test_age_days(self):
        v = SmartVector(created_at=datetime.now() - timedelta(days=10))
        assert 9.9 < v.age_days < 10.1
