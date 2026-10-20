# MetaFeed AI

An end-to-end recommendation and ML systems engineering platform.

## Purpose

Explore technologies and engineering concepts relevant to
Meta's recommendation infrastructure, including:

- React and TypeScript
- Swift and SwiftUI
- Python and FastAPI
- PyTorch and TorchRec
- FAISS vector retrieval
- DLRM-inspired ranking
- Recommendation evaluation and inference optimization

## Milestone 1 — Popularity Baseline

Implemented a deterministic recommendation API using
a small synthetic fixture.

Algorithm:
1. Count positive interactions for every item.
2. Exclude items already positively interacted with by the user.
3. Rank by descending popularity.
4. Resolve ties using ascending item IDs.
5. Return the requested number of recommendations.

This baseline does not perform learned personalization.

## Development

Create and activate a Python virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -e ".[dev]"

Run tests:

    python -m pytest -v

Start API:

    python -m uvicorn metafeed.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs

Example endpoint:

    GET /v1/users/alice/recommendations?limit=10

## Real-world dataset

KuaiRec 2.0 is used for future recommendation experiments.

Initial local audit:
- 12,530,806 interaction records
- 7,176 unique users
- 10,728 unique videos
- 0 missing values detected
- 0 invalid numeric rows detected

Raw data is excluded from Git.

## Planned engineering work

- KuaiRec ingestion and chronological evaluation
- PyTorch recommendation models
- FAISS retrieval
- DLRM-inspired ranking
- Isolated TorchRec experiments
- React web application
- Native SwiftUI iOS application
- Inference benchmarking and reliability testing

## Limitations

- Current API uses only synthetic test data.
- No trained ML models are deployed.
- No production-scale performance claims.
