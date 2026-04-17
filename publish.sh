#!/bin/bash
# SmartVector — Publish to GitHub
# Run this script from the smartvector directory

set -e

echo "=== SmartVector GitHub Publisher ==="

# Clean up any stale git state
rm -rf .git

# Initialize fresh git repo
git init
git branch -m main
git add -A

git commit -m "Initial release: SmartVector — self-aware vector embeddings for RAG

Self-aware vector embeddings with temporal awareness, confidence decay,
and relational intelligence. Solves the RAG versioning problem where
traditional vector databases cannot distinguish outdated, conflicting,
or low-trust information.

Key components:
- SmartVector: dataclass with temporal, confidence, and graph metadata
- ConfidenceEngine: Ebbinghaus-inspired exponential decay + feedback
- ConsolidationAgent: background conflict detection and ripple propagation
- SmartVectorDB: 4-signal retrieval (similarity + temporal + confidence + relational)
- Surgical update: diff-based chunk replacement via source-offset mapping

63 tests, CLI demo, examples, and interactive theory documentation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

echo ""
echo "Git repo initialized with commit."
echo ""

# Create GitHub repo and push
if command -v gh &> /dev/null; then
    echo "Found gh CLI. Creating GitHub repo..."
    gh repo create smartvector --public \
        --description "Self-aware vector embeddings for RAG — temporal awareness, confidence decay, relational intelligence" \
        --source . --push
    echo ""
    echo "=== Published! ==="
    echo "Visit: https://github.com/$(gh api user -q .login)/smartvector"
else
    echo "gh CLI not found. Trying git push..."
    echo "Please create repo at https://github.com/new first, then:"
    echo "  git remote add origin https://github.com/xunaizhong/smartvector.git"
    echo "  git push -u origin main"
fi
