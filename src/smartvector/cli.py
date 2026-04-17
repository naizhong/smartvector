"""
CLI entry point for SmartVector.

Usage::

    smartvector demo          # Run the full lifecycle demo
    smartvector --version     # Show version
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="smartvector",
        description="SmartVector — self-aware vector embeddings for RAG",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run the full lifecycle demo")

    args = parser.parse_args(argv)

    if args.version:
        from smartvector import __version__
        print(f"smartvector {__version__}")
        return

    if args.command == "demo":
        _run_demo()
    else:
        parser.print_help()


def _run_demo() -> None:
    """Run the end-to-end Smart Vector lifecycle demo."""
    from datetime import datetime, timedelta
    from smartvector import (
        SmartVectorDB,
        VectorStatus,
    )

    def banner(text: str) -> None:
        print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")

    banner("SMART VECTOR — FULL LIFECYCLE DEMO")
    db = SmartVectorDB()

    # === Stage 1: Encoding ===
    banner("STAGE 1: ENCODING — Fast Ingest")

    tech_doc = (
        "Section 1: System Overview. "
        "The platform processes 10,000 requests per second. "
        "The sky color calibration uses a blue baseline for atmospheric modeling. "
        "All sensors are calibrated quarterly.\n"
        "Section 2: Pricing. "
        "Enterprise plan costs $99 per month per seat. "
        "Volume discounts available for teams over 50 seats. "
        "Annual billing gives 20% discount."
    )
    results = db.ingest_document(
        doc_id="tech-spec", version=1, text=tech_doc,
        source_name="engineering_wiki", source_type="technical_doc",
        author="alice", authority=0.85,
        created_at=datetime.now() - timedelta(days=30),
    )
    print(f"  Ingested tech-spec v1: {len(results)} chunks")
    for r in results:
        vid = r.get("vector_id", r.get("new_vector", ""))
        print(f"    {r['action']}: {vid}")

    slack_msg = "I heard the sky calibration is switching to yellow baseline next quarter."
    db.ingest_document(
        doc_id="slack-rumor", version=1, text=slack_msg,
        source_name="slack_engineering", source_type="slack",
        author="intern_bob", authority=0.30,
        created_at=datetime.now() - timedelta(days=5),
    )
    print(f"  Ingested slack rumor (authority=0.30)")

    hr_doc = (
        "Remote work policy: employees must be in-office 3 days per week "
        "(Mon/Wed/Fri). Remote permitted Tue/Thu."
    )
    db.ingest_document(
        doc_id="hr-policy", version=1, text=hr_doc,
        source_name="hr_policy_db", source_type="policy_doc",
        author="hr_director", authority=0.95,
        created_at=datetime.now() - timedelta(days=10),
    )
    print(f"  Ingested HR policy (authority=0.95)")
    print(f"\n  DB Stats: {db.stats()}")

    # === Stage 2: Consolidation ===
    banner("STAGE 2: CONSOLIDATION — Background Processing")
    cr = db.run_consolidation()
    print(f"  Vectors processed: {cr['vectors_processed']}")
    print(f"  Conflicts found: {cr['conflicts_found']}")
    print(f"  Relationships built: {cr['relationships_built']}")
    print(f"  Vectors consolidated: {cr['vectors_consolidated']}")

    # === Stage 3: Retrieval ===
    banner("STAGE 3: RETRIEVAL & REINFORCEMENT")
    print("\n  Query: 'What color baseline does the sky calibration use?'")
    results = db.query("What color baseline does the sky calibration use?", top_k=3)
    for i, r in enumerate(results):
        marker = " <-- WINNER" if i == 0 else ""
        print(f"\n  Result {i + 1}{marker}:")
        print(f"    Content: \"{r.vector.content[:70]}...\"")
        print(f"    Source: {r.vector.source_name} | Author: {r.vector.author}")
        print(f"    Scores: sim={r.similarity_score}, temp={r.temporal_score}, "
              f"conf={r.confidence_score}, rel={r.relational_bonus}")
        print(f"    FINAL: {r.final_score}")

    winner = results[0]
    db.record_feedback(winner.vector.vector_id, accepted=True)
    print(f"\n  User accepted answer -> confidence boosted")

    # === Stage 4: Surgical Update ===
    banner("STAGE 4: SURGICAL UPDATE + RIPPLE PROPAGATION")
    updated_doc = tech_doc.replace(
        "blue baseline", "yellow baseline"
    ).replace(
        "$99 per month", "$149 per month"
    )
    print("  Changes: 'blue baseline' -> 'yellow baseline', '$99' -> '$149'")

    ur = db.ingest_update(
        doc_id="tech-spec", old_text=tech_doc, new_text=updated_doc,
        source_name="engineering_wiki", source_type="technical_doc",
        author="alice_v2", authority=0.90,
    )
    print(f"  Chunks affected: {ur['chunks_affected']}, updated: {ur['chunks_updated']}")
    db.run_consolidation()

    print("\n  Query after update: 'What color baseline?'")
    results_after = db.query("What color baseline does the sky calibration use?", top_k=2)
    for i, r in enumerate(results_after):
        marker = " <-- WINNER" if i == 0 else ""
        print(f"\n  Result {i + 1}{marker}:")
        print(f"    Content: \"{r.vector.content[:70]}...\"")
        print(f"    Version: v{r.vector.doc_version} | FINAL: {r.final_score}")

    # === Stage 5: Decay ===
    banner("STAGE 5: DECAY & DORMANCY — 180 days later")
    future = datetime.now() + timedelta(days=180)
    results_future = db.query("What color baseline?", top_k=2, reference_time=future)
    for i, r in enumerate(results_future):
        print(f"\n  Result {i + 1}:")
        print(f"    Content: \"{r.vector.content[:60]}...\"")
        print(f"    Confidence (decayed): {r.confidence_score}")
        print(f"    FINAL: {r.final_score}")

    # === Summary ===
    banner("FINAL STATE")
    stats = db.stats()
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  By status: {stats['by_status']}")
    print(f"  Avg confidence (active): {stats['avg_confidence']}")
    print("\n  Done! All 5 lifecycle stages demonstrated.")


if __name__ == "__main__":
    main()
