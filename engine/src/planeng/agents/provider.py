"""Which model answers, and where it runs.

Two backends, one interface. The agent above this line does not know or care
which is in use, and that matters more than it sounds: the whole argument for
this design is that the language model is a replaceable part. If swapping it
meant editing the agent, it would not be replaceable, it would be load bearing.

Selection order:

1. `PLANENG_LLM_PROVIDER`, if set to `groq`, `ollama` or `none`.
2. Otherwise Groq, if a key is present and the endpoint answers.
3. Otherwise a local Ollama, if one is running.
4. Otherwise nothing, and the caller falls back to the rule based reader.

Automatic rather than fixed because the two deployments genuinely differ. On a
planner's workstation a local model costs nothing per brief and keeps the brief
in the building. On a hosted demo there is no GPU and no model to pull, so the
same code has to reach a hosted endpoint instead. Neither is correct in general.

The key is read from the environment, and a `.env` beside the project is loaded
into it if present. It is never read from a file the engine ships, never logged
and never returned by the status endpoint, which reports only whether a key is
present.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from . import groq, ollama

_ENV_LOADED = False


def load_env(start: pathlib.Path | None = None) -> None:
    """Read a .env from this directory or any parent, once.

    Deliberately tiny rather than a dependency: this reads `KEY=value` lines and
    nothing else, and it never overwrites a variable that is already set, so a
    real secret store on the host always wins over a file left on disk.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"\''))
        return


@dataclass(frozen=True)
class Backend:
    """One way of reaching a model."""

    name: str
    module: object
    model: str
    where: str

    @property
    def error(self) -> type[Exception]:
        return self.module.LLMError if hasattr(self.module, "LLMError") else (
            groq.GroqError if self.name == "groq" else ollama.OllamaError)

    def chat_json(self, messages, schema, **kwargs):
        kwargs.setdefault("model", self.model)
        return self.module.chat_json(messages, schema, **kwargs)


def select(prefer: str | None = None) -> Backend | None:
    """The backend to use now, or None if no model is reachable."""
    load_env()
    choice = (prefer or os.environ.get("PLANENG_LLM_PROVIDER", "")).strip().lower()

    if choice == "none":
        return None
    if choice == "groq":
        return _groq() if groq.available() else None
    if choice == "ollama":
        return _ollama() if ollama.available() else None

    # No preference stated: whichever is actually there, cloud first, because a
    # machine with both is usually a developer machine and the hosted path is
    # the one the client will be using.
    if groq.available():
        return _groq()
    if ollama.available():
        return _ollama()
    return None


def status() -> dict:
    """What the AI layer is running on, for the interface to state plainly.

    Reports whether a key is present, never the key.
    """
    load_env()
    backend = select()
    return {
        "reachable": backend is not None,
        "provider": backend.name if backend else None,
        "default_model": backend.model if backend else None,
        "where": backend.where if backend else "no model reachable",
        "mode": f"{backend.where} model" if backend else "rule based reader",
        "groq_key_present": bool(groq.api_key()),
        "ollama_reachable": ollama.available(),
        "installed_models": (backend.module.installed_models()
                             if backend else []),
    }


def _groq() -> Backend:
    return Backend(name="groq", module=groq, model=groq.DEFAULT_MODEL,
                   where="hosted")


def _ollama() -> Backend:
    return Backend(name="ollama", module=ollama, model=ollama.DEFAULT_MODEL,
                   where="local")
