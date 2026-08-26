"""Test-wide defaults.

The agent can now reach a hosted model, which makes the test suite a network
client unless something stops it. That is unacceptable in two directions: the
suite would be slow and flaky on a bad connection, and it would spend a real
API allowance every time anyone runs it. Worse, it would pass or fail depending
on whether a key happened to be present in the environment.

So no model is reachable by default. Tests that want a model stub the provider
explicitly, which is the honest way round: a test that needs a language model
should say so in the test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """Keep the suite off the network and off anyone's API allowance."""
    monkeypatch.setenv("PLANENG_LLM_PROVIDER", "none")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
