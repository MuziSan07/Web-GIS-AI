"""Tests for backend selection and the hosted transport.

Two things here are worth guarding rather than trusting.

The first is that a key never leaves by accident. The status endpoint is
public, the interface prints what it returns, and a key pasted into a chat
window is already compromised - so the test asserts on the absence of the
secret, not on the presence of a flag.

The second is that structured output is genuinely structured. A hosted API will
happily return valid JSON with invented field names if asked the wrong way, and
that failure is invisible until a plan is built from fields nothing reads.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from planeng.agents import groq, ollama, provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("PLANENG_LLM_PROVIDER", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(provider, "_ENV_LOADED", True)   # do not read a real .env
    yield


# -- selection --------------------------------------------------------------

def test_none_means_none(monkeypatch):
    """A caller measuring the deterministic path must not be handed a model."""
    monkeypatch.setenv("PLANENG_LLM_PROVIDER", "none")
    monkeypatch.setattr(groq, "available", lambda *a, **k: True)
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)
    assert provider.select() is None


def test_explicit_choice_is_honoured_even_when_both_are_up(monkeypatch):
    monkeypatch.setattr(groq, "available", lambda *a, **k: True)
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)

    monkeypatch.setenv("PLANENG_LLM_PROVIDER", "ollama")
    assert provider.select().name == "ollama"

    monkeypatch.setenv("PLANENG_LLM_PROVIDER", "groq")
    assert provider.select().name == "groq"


def test_an_explicit_choice_that_is_down_does_not_fall_through(monkeypatch):
    """Asking for one backend and silently getting another hides an outage."""
    monkeypatch.setenv("PLANENG_LLM_PROVIDER", "groq")
    monkeypatch.setattr(groq, "available", lambda *a, **k: False)
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)
    assert provider.select() is None


def test_hosted_wins_when_nothing_is_stated(monkeypatch):
    monkeypatch.setattr(groq, "available", lambda *a, **k: True)
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)
    assert provider.select().name == "groq"


def test_falls_back_to_local_then_to_nothing(monkeypatch):
    monkeypatch.setattr(groq, "available", lambda *a, **k: False)
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)
    assert provider.select().name == "ollama"

    monkeypatch.setattr(ollama, "available", lambda *a, **k: False)
    assert provider.select() is None


# -- secrets ----------------------------------------------------------------

def test_status_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_thisisnotarealkey_000")
    monkeypatch.setattr(groq, "available", lambda *a, **k: True)
    monkeypatch.setattr(groq, "installed_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(ollama, "available", lambda *a, **k: False)

    body = json.dumps(provider.status())
    assert "gsk_" not in body
    assert "thisisnotarealkey" not in body
    assert json.loads(body)["groq_key_present"] is True


def test_no_key_means_the_hosted_backend_is_not_available(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert groq.api_key() is None
    assert groq.available() is False


def test_env_file_never_overrides_the_real_environment(monkeypatch, tmp_path):
    """A stale .env on a server must not beat the host's secret store."""
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-the-file\n",
                                   encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "from-the-host")
    monkeypatch.setattr(provider, "_ENV_LOADED", False)
    provider.load_env(tmp_path / "anything.py")
    assert groq.api_key() == "from-the-host"


# -- the hosted transport ---------------------------------------------------

def test_a_schema_is_sent_as_a_schema_not_as_loose_json(monkeypatch):
    """`json_object` promises valid JSON and nothing about the fields.

    Asked for a planning brief that way, the model returned a density per square
    kilometre and a land use split that nothing downstream reads, and left out
    fields that are required.
    """
    sent = {}

    def fake(method, path, body=None, timeout=30.0):
        sent.update(body or {})
        return {"choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"completion_tokens": 3}, "model": "stub"}

    monkeypatch.setattr(groq, "_request", fake)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    result, completion = groq.chat_json([{"role": "user", "content": "hi"}],
                                        schema)

    assert result == {"ok": True}
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert sent["temperature"] == 0        # extraction, not composition
    assert completion.eval_tokens == 3


def test_text_that_is_not_json_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setattr(groq, "_request", lambda *a, **k: {
        "choices": [{"message": {"content": "Sure! Here is your brief:"}}]})
    with pytest.raises(groq.GroqError):
        groq.chat_json([{"role": "user", "content": "hi"}], {"type": "object"})


def test_a_rate_limit_is_waited_out_not_failed(monkeypatch):
    """The free tier meters tokens a minute and refills in about twenty
    seconds. Failing a planner over that would be absurd."""
    calls = {"n": 0}
    slept: list[float] = []

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "u", 429, "Too Many Requests",
                {"retry-after": "2"}, None)
        return {"choices": [{"message": {"content": "{}"}}], "usage": {}}

    monkeypatch.setattr(groq, "_request", flaky)
    monkeypatch.setattr(groq.time, "sleep", lambda s: slept.append(s))

    groq.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2
    assert slept == [2.0]


def test_a_real_error_is_not_retried(monkeypatch):
    """Retrying a bad key just makes the planner wait for the same answer."""
    calls = {"n": 0}

    def unauthorised(*a, **k):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {},
                                     __import__("io").BytesIO(b"bad key"))

    monkeypatch.setattr(groq, "_request", unauthorised)
    with pytest.raises(groq.GroqError):
        groq.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_reset_hints_are_read_in_seconds_and_minutes():
    def hint(value):
        return urllib.error.HTTPError(
            "u", 429, "x", {"x-ratelimit-reset-tokens": value}, None)

    assert groq._retry_after(hint("19.117s")) == pytest.approx(19.117)
    assert groq._retry_after(hint("1m4s")) == pytest.approx(30.0)   # capped
    assert groq._retry_after(hint("")) == 5.0                       # default
