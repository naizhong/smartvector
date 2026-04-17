# SmartVector

**Self-aware vector embeddings for RAG systems that know _what_ they mean, _when_ they're valid, _how_ confident they should be, and _who_ they're related to.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)](#testing)

---

## The Problem

Traditional RAG systems treat every vector embedding as equal — a fact from an authoritative database and a rumor from Slack get the same treatment. When documents are updated, old chunks linger in the vector store with no way to know they're outdated. When two sources contradict each other, the system has no mechanism to prefer one over the other.

SmartVector solves this by giving each vector three properties that traditional embeddings lack:

1. **Temporal Awareness** — vectors know when they were created, when they expire, and how their relevance decays over time
2. **Confidence Decay** — vectors carry a trust score derived from source authority, user feedback, and age, modeled on the Ebbinghaus forgetting curve
3. **Relational Awareness** — vectors form a knowledge graph with dependency edges, enabling ripple propagation when upstream facts change

## Installation

```bash
pip install smartvector
```

Or install from source:

```bash
git clone https://github.com/xunaizhong/smartvector.git
cd smartvector
pip install -e ".[dev]"
```

## Quick Start

```python
from smartvector import SmartVectorDB

db = SmartVectorDB()

# Ingest a document (vectors start as UNCONSOLIDATED)
db.ingest_document(
    doc_id="api-spec", version=1,
    text="The API rate limit is 1000 requests per second.",
    source_name="api_docs", source_type="technical_doc",
    author="alice", authority=0.85,
)

# Run consolidation (like the brain's sleep cycle)
db.run_consolidation()

# Query with 4-signal scoring
results = db.query("What is the rate limit?", top_k=3)
print(results[0].vector.content)
# → "The API rate limit is 1000 requests per second."
print(results[0].final_score)
# → 0.7523 (similarity + temporal + confidence + relational)

# Surgical update — only affected chunks are replaced
db.ingest_update(
    doc_id="api-spec",
    old_text="The API rate limit is 1000 requests per second.",
    new_text="The API rate limit is 2000 requests per second.",
    source_name="api_docs", source_type="technical_doc",
    author="alice_v2", authority=0.90,
)

# Old vectors are DEPRECATED, new ones take over
results = db.query("What is the rate limit?", top_k=1)
print(results[0].vector.content)
# → "The API rate limit is 2000 requests per second."
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SmartVectorDB                         │
│                                                          │
│  ┌────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │ Vector     │  │ Confidence     │  │ Consolidation │ │
│  │ Store      │──│ Engine         │──│ Agent         │ │
│  └────────────┘  └────────────────┘  └───────────────┘ │
│       │                │                    │            │
│  ┌────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │ Temporal   │  │ 4-Signal       │  │ Ripple        │ │
│  │ Scorer     │──│ Retrieval      │──│ Propagator    │ │
│  └────────────┘  └────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 4-Signal Retrieval Scoring

```
final = 0.35 × similarity
      + 0.25 × temporal_score
      + 0.25 × confidence
      + 0.15 × relational_bonus
```

Each signal is configurable. In production, replace the keyword similarity with embedding cosine similarity.

### 5 Lifecycle Stages (Neuroscience-Inspired)

| Stage | Name | Analogy | What Happens |
|-------|------|---------|--------------|
| 1 | **Encoding** | Hippocampal fast-write | Document ingested, vectors created as UNCONSOLIDATED |
| 2 | **Consolidation** | Sleep-time replay | Background agent builds relationships, detects conflicts |
| 3 | **Retrieval** | Memory recall | 4-signal scoring, user feedback reinforces confidence |
| 4 | **Update** | Schema revision | Surgical diff, ripple propagation through knowledge graph |
| 5 | **Decay** | Forgetting curve | Confidence decays exponentially, low-trust → DORMANT |

## Key Components

### SmartVector

The fundamental unit — a dataclass carrying content, temporal metadata, confidence signals, and graph edges.

### ConfidenceEngine

Manages the trust lifecycle: exponential decay (`C(t) = C₀ × 2^(-t/half_life)`), feedback adjustment, and access reinforcement.

### ConsolidationAgent

The "sleep-time" background process that detects conflicts, builds relationship edges, and propagates ripples when facts change.

### SmartVectorDB

The complete database combining ingestion, consolidation, 4-signal retrieval, surgical updates, and LLM context building.

## Source Authority Hierarchy

| Source Type | Default Confidence |
|-------------|-------------------|
| Official DB (CRM, ERP) | 0.95 |
| Policy Documents | 0.90 |
| Technical Docs | 0.85 |
| Wiki / Confluence | 0.75 |
| Email | 0.50 |
| Meeting Notes | 0.45 |
| Slack Messages | 0.30 |
| Unknown | 0.20 |

## Examples

Run the built-in demo:

```bash
smartvector demo
```

Or run individual examples:

```bash
python examples/quickstart.py
python examples/conflict_resolution.py
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

63 tests covering all components: models, confidence engine, conflict detection, ripple propagation, surgical updates, and full lifecycle.

## Theory & Research

SmartVector draws from:

- **Neuroscience**: Hippocampal-neocortical memory consolidation, Ebbinghaus forgetting curve, memory reconsolidation
- **Temporal Knowledge Graphs**: DE-SimplE, ATiSE, PTBox — time-aware embeddings
- **MAGMA Architecture**: Dual-stream (fast ingestion + slow consolidation)
- **GNN Message Passing**: Ripple propagation through knowledge graphs
- **VersionRAG**: 90% accuracy on versioned queries vs 58% standard RAG

See [`docs/theory_visual.html`](docs/theory_visual.html) for the full interactive proposal with implementation details.

## Roadmap

**Phase 1 (Current)**: Core data structures, confidence engine, consolidation agent, keyword-based retrieval, surgical updates, CLI demo.

**Phase 2 (Next)**: Real embedding integration (sentence-transformers), persistent storage (SQLite/Postgres), NLI-based conflict detection, REST API.

**Phase 3 (Future)**: Distributed consolidation, real-time streaming ingestion, Matryoshka adaptive embeddings, production integrations (LangChain, LlamaIndex).

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-thing`)
3. Make your changes with tests
4. Run `pytest tests/ -v` to verify
5. Open a Pull Request

## License

MIT — see [LICENSE](LICENSE) for details.
