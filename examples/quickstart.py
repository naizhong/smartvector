"""
SmartVector Quickstart
======================
A minimal example showing the core workflow:
ingest → consolidate → query → update → query again.
"""

from smartvector import SmartVectorDB

# 1. Create a database
db = SmartVectorDB()

# 2. Ingest some documents
db.ingest_document(
    doc_id="api-spec",
    version=1,
    text=(
        "The API rate limit is 1000 requests per second. "
        "Authentication requires a Bearer token in the Authorization header. "
        "All responses are JSON-encoded."
    ),
    source_name="api_docs",
    source_type="technical_doc",
    author="alice",
    authority=0.85,
)

db.ingest_document(
    doc_id="slack-note",
    version=1,
    text="I think the rate limit might be changing to 500 req/s next sprint.",
    source_name="slack",
    source_type="slack",
    author="intern_bob",
    authority=0.30,
)

# 3. Run consolidation (background process)
result = db.run_consolidation()
print(f"Consolidated: {result['vectors_consolidated']} vectors, "
      f"{result['conflicts_found']} conflicts found")

# 4. Query — the high-authority doc wins
results = db.query("What is the API rate limit?", top_k=2)
for i, r in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(f"  Content: {r.vector.content[:80]}...")
    print(f"  Source: {r.vector.source_name} (authority={r.vector.base_confidence})")
    print(f"  Score: {r.final_score}")
    if r.conflict_warnings:
        print(f"  Warnings: {r.conflict_warnings}")

# 5. Record user feedback
db.record_feedback(results[0].vector.vector_id, accepted=True)

# 6. Surgical update — change rate limit from 1000 to 2000
old_text = (
    "The API rate limit is 1000 requests per second. "
    "Authentication requires a Bearer token in the Authorization header. "
    "All responses are JSON-encoded."
)
new_text = old_text.replace("1000 requests", "2000 requests")

update = db.ingest_update(
    doc_id="api-spec",
    old_text=old_text,
    new_text=new_text,
    source_name="api_docs",
    source_type="technical_doc",
    author="alice_v2",
    authority=0.90,
)
print(f"\nUpdate: {update['chunks_affected']} chunks affected, "
      f"now at v{update['new_version']}")

# 7. Query again — should get the updated answer
results = db.query("What is the API rate limit?", top_k=1)
print(f"\nAfter update: {results[0].vector.content[:80]}...")

# 8. Build LLM context
context = db.build_llm_context("What is the rate limit?")
print(f"\n--- LLM Context ---\n{context}")
