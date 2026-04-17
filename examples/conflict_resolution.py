"""
Conflict Resolution Example
============================
Shows how SmartVector handles conflicting information from
multiple sources with different authority levels.
"""

from datetime import datetime, timedelta
from smartvector import SmartVectorDB

db = SmartVectorDB()

# Scenario: Three sources say different things about the same topic

# Source 1: Official database (high authority)
db.ingest_document(
    doc_id="official-db",
    version=1,
    text="The company headquarters is located in San Francisco, California.",
    source_name="company_database",
    source_type="official_db",
    author="system",
    authority=0.95,
    created_at=datetime.now() - timedelta(days=60),
)

# Source 2: Wiki article (medium authority, more recent)
db.ingest_document(
    doc_id="wiki-article",
    version=1,
    text="The company headquarters recently moved to Austin, Texas.",
    source_name="internal_wiki",
    source_type="wiki",
    author="marketing_team",
    authority=0.75,
    created_at=datetime.now() - timedelta(days=5),
)

# Source 3: Slack message (low authority, most recent)
db.ingest_document(
    doc_id="slack-msg",
    version=1,
    text="Heard we might be moving HQ to New York next year, not confirmed.",
    source_name="slack_general",
    source_type="slack",
    author="random_employee",
    authority=0.30,
    created_at=datetime.now() - timedelta(hours=2),
)

# Run consolidation to detect conflicts
result = db.run_consolidation()
print(f"Conflicts detected: {result['conflicts_found']}")
print(f"Relationships built: {result['relationships_built']}")

# Query
print("\n--- Query: 'Where is the company headquarters?' ---")
results = db.query("Where is the company headquarters?", top_k=3)

for i, r in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(f"  Content: {r.vector.content}")
    print(f"  Source: {r.vector.source_name} | Author: {r.vector.author}")
    print(f"  Scores:")
    print(f"    Similarity: {r.similarity_score}")
    print(f"    Temporal:   {r.temporal_score}")
    print(f"    Confidence: {r.confidence_score}")
    print(f"    Relational: {r.relational_bonus}")
    print(f"    FINAL:      {r.final_score}")
    if r.conflict_warnings:
        print(f"  Warnings:")
        for w in r.conflict_warnings:
            print(f"    - {w}")

# Show LLM context (what the AI actually sees)
print("\n\n--- LLM Context ---")
print(db.build_llm_context("Where is the company headquarters?"))
