"""
Shared pytest fixtures for the NeuroSeek test suite.

The embedding model takes several seconds to load from disk.  By loading it
once per test session and injecting the resulting Embedder into every test
that needs one, we avoid paying that cost hundreds of times.
"""

import pytest
from neuroseek.embedder import Embedder


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """A single Embedder instance shared across the entire test session."""
    return Embedder()
