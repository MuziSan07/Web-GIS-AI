"""Ollama transport.

A local model runs the agent layer, so a planner can use the system without an
API key, a network connection, or a per-token bill. The engine never depends on
which model answers - the agent's only output is a PlanningBrief, and that is
validated before anything downstream sees it.

Deliberately dependency-free: the stdlib can do an HTTP POST, and this is the
one module that would otherwise drag a client library into a geospatial engine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("PLANENG_LLM_MODEL", "llama3.2:3b")

# Some GPUs are advertised by the runtime but cannot actually serve a model -
# older Pascal cards fail the Vulkan 16-bit storage check and take the whole
# inference subprocess down with them. Rather than hardcode a machine quirk,
# the first GPU failure flips this latch and every later call runs on CPU.
_force_cpu = os.environ.get("PLANENG_LLM_CPU", "").lower() in ("1", "true", "yes")

# Errors that mean "the GPU could not serve this", not "the request was wrong".
_GPU_FAILURE_MARKERS = (
    "llama-server process has terminated",
    "16-bit storage",
    "vulkan",
    "cuda",
    "out of memory",
)


class OllamaError(RuntimeError):
    """Ollama could not answer. Callers fall back rather than failing the request."""


@dataclass
class Completion:
    text: str
    model: str
    duration_s: float
    eval_tokens: int
    ran_on_cpu: bool          # whether we *sent* num_gpu=0, not where it ran

    @property
    def tokens_per_second(self) -> float:
        return self.eval_tokens / self.duration_s if self.duration_s > 0 else 0.0


def available(host: str = DEFAULT_HOST, timeout: float = 2.0) -> bool:
    """Is a local model server reachable? Used to decide whether to even try."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def installed_models(host: str = DEFAULT_HOST, timeout: float = 5.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return []


def chat(
    messages: list[dict],
    *,
    schema: dict | None = None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 900.0,
    temperature: float = 0.0,
) -> Completion:
    """One completion, optionally constrained to a JSON schema.

    Passing a schema is not a formatting nicety: it is what stops a small model
    inventing field names. The server constrains decoding to the grammar, so
    the reply parses as JSON or the call fails - there is no regex salvage.

    Temperature defaults to 0. A planning brief is an extraction task; sampling
    variety here means the same prompt yields different populations.
    """
    global _force_cpu

    def _post(cpu_only: bool) -> dict:
        options = {"temperature": temperature}
        if cpu_only:
            options["num_gpu"] = 0
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if schema is not None:
            body["format"] = schema

        req = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    attempts = [True] if _force_cpu else [False, True]
    last_error: str = ""

    for cpu_only in attempts:
        try:
            payload = _post(cpu_only)
        except urllib.error.HTTPError as exc:
            last_error = exc.read().decode(errors="replace")[:400]
            if not cpu_only and _looks_like_gpu_failure(last_error):
                # Latch to CPU so every later call skips the doomed GPU attempt.
                _force_cpu = True
                continue
            raise OllamaError(
                f"Ollama returned HTTP {exc.code}: {last_error}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Could not reach Ollama at {host}. Is `ollama serve` running? ({exc.reason})"
            ) from exc

        return Completion(
            text=payload["message"]["content"],
            model=payload.get("model", model),
            duration_s=payload.get("total_duration", 0) / 1e9,
            eval_tokens=payload.get("eval_count", 0),
            ran_on_cpu=cpu_only,
        )

    raise OllamaError(f"Ollama failed on both GPU and CPU: {last_error}")


def _looks_like_gpu_failure(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in _GPU_FAILURE_MARKERS)


def chat_json(messages: list[dict], schema: dict, **kwargs) -> tuple[dict, Completion]:
    """Constrained completion, parsed. Raises OllamaError if it isn't valid JSON."""
    completion = chat(messages, schema=schema, **kwargs)
    try:
        return json.loads(completion.text), completion
    except json.JSONDecodeError as exc:
        raise OllamaError(
            f"Model returned text that is not JSON despite a schema constraint: "
            f"{completion.text[:200]}"
        ) from exc
