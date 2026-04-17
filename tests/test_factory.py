"""Tests for SmartVectorFactory."""

from datetime import datetime

from smartvector.factory import SmartVectorFactory
from smartvector.models import VectorStatus


class TestSmartVectorFactory:
    def test_basic_chunking(self):
        text = "First sentence. Second sentence. Third sentence."
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text=text,
            source_name="test", source_type="wiki", author="alice",
        )
        assert len(vectors) >= 1
        assert all(v.doc_id == "doc-1" for v in vectors)
        assert all(v.doc_version == 1 for v in vectors)
        assert all(v.status == VectorStatus.UNCONSOLIDATED for v in vectors)

    def test_source_offsets_set(self):
        text = "Sentence one here. Sentence two here. Sentence three here."
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text=text,
            source_name="test", source_type="wiki", author="alice",
        )
        for v in vectors:
            assert v.source_offset_start >= 0
            assert v.source_offset_end > v.source_offset_start

    def test_chunk_indices_sequential(self):
        text = (
            "A very long first sentence that goes on and on. "
            "Another equally verbose sentence here. "
            "Yet another sentence to fill the chunk. "
            "And more content to trigger a second chunk hopefully. "
            "Extra sentences are always welcome in tests. "
            "The final sentence wraps it up nicely here."
        )
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text=text,
            source_name="test", source_type="wiki", author="alice",
            chunk_size=100,
        )
        indices = [v.chunk_index for v in vectors]
        assert indices == list(range(len(vectors)))

    def test_authority_defaults_to_unknown(self):
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text="Hello world.",
            source_name="test", source_type="unknown", author="bob",
        )
        assert vectors[0].base_confidence == 0.20  # SourceAuthority.UNKNOWN

    def test_custom_authority(self):
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text="Hello world.",
            source_name="test", source_type="wiki", author="bob",
            authority=0.95,
        )
        assert vectors[0].base_confidence == 0.95

    def test_custom_created_at(self):
        ts = datetime(2025, 1, 1, 12, 0, 0)
        vectors = SmartVectorFactory.from_text(
            doc_id="doc-1", version=1, text="Hello world.",
            source_name="test", source_type="wiki", author="bob",
            created_at=ts,
        )
        assert vectors[0].created_at == ts
