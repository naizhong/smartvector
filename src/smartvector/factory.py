"""
SmartVector Factory — creates SmartVectors from raw text.

Handles chunking with source-offset tracking so that when a document
is later edited, the system can identify exactly which chunks are
affected by the change.
"""

from __future__ import annotations

from datetime import datetime

from smartvector.models import SmartVector, SourceAuthority


class SmartVectorFactory:
    """Creates :class:`SmartVector` instances from raw text.

    The factory splits text into sentence-based chunks and records the
    character offsets of each chunk relative to the original document.
    This enables *surgical updates* — when a document changes, only the
    affected chunks are replaced.
    """

    @staticmethod
    def from_text(
        doc_id: str,
        version: int,
        text: str,
        source_name: str,
        source_type: str,
        author: str,
        authority: float | None = None,
        created_at: datetime | None = None,
        chunk_size: int = 200,
    ) -> list[SmartVector]:
        """Chunk *text* and create SmartVectors with full provenance.

        Parameters
        ----------
        doc_id : str
            Identifier for the source document.
        version : int
            Document version number.
        text : str
            The raw document text.
        source_name, source_type, author : str
            Provenance metadata.
        authority : float, optional
            Base confidence (defaults to :attr:`SourceAuthority.UNKNOWN`).
        created_at : datetime, optional
            Timestamp for the vectors (defaults to *now*).
        chunk_size : int
            Target chunk size in characters.

        Returns
        -------
        list[SmartVector]
            One vector per chunk, with ``source_offset_start`` and
            ``source_offset_end`` set for surgical-update support.
        """
        if authority is None:
            authority = SourceAuthority.UNKNOWN.value
        if created_at is None:
            created_at = datetime.now()

        # Split into sentences
        sentences: list[str] = []
        current = ""
        for char in text:
            current += char
            if char in ".!?\n" and len(current.strip()) > 5:
                sentences.append(current)
                current = ""
        if current.strip():
            sentences.append(current)

        # Group into chunks respecting sentence boundaries
        vectors: list[SmartVector] = []
        current_chunk = ""
        current_start = 0
        offset = 0
        chunk_idx = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                vectors.append(SmartVector(
                    doc_id=doc_id,
                    doc_version=version,
                    chunk_index=chunk_idx,
                    content=current_chunk.strip(),
                    source_name=source_name,
                    source_type=source_type,
                    author=author,
                    base_confidence=authority,
                    created_at=created_at,
                    updated_at=created_at,
                    source_offset_start=current_start,
                    source_offset_end=current_start + len(current_chunk),
                    half_life_days=30.0,
                ))
                chunk_idx += 1
                current_start = offset
                current_chunk = sentence
            else:
                current_chunk += sentence
            offset += len(sentence)

        if current_chunk.strip():
            vectors.append(SmartVector(
                doc_id=doc_id,
                doc_version=version,
                chunk_index=chunk_idx,
                content=current_chunk.strip(),
                source_name=source_name,
                source_type=source_type,
                author=author,
                base_confidence=authority,
                created_at=created_at,
                updated_at=created_at,
                source_offset_start=current_start,
                source_offset_end=current_start + len(current_chunk),
                half_life_days=30.0,
            ))

        return vectors
