"""Groq transport, for when the model runs in the cloud rather than on the desk.

The local Ollama path is the right default when a planner has a workstation: no
per call cost, and briefs never leave the building. It is the wrong default for
a hosted demo, where there is no GPU, no model to pull and a cold container that
must answer in a second. This is the same agent talking to a hosted endpoint
instead.

Schema constrained decoding is kept, and that is the whole reason this file
exists rather than a generic OpenAI client. Groq offers two structured modes and
they are not equivalent:

* `json_schema` constrains generation to the schema. Asked for a planning brief
  it returns exactly the fields the brief has.
* `json_object` only promises valid JSON. Asked the same question it returned
  `density_per_km2` and a `land_use` breakdown that nothing downstream reads,
  and omitted fields that are required. Free form JSON is not structured output.

So `json_schema` is used, and a model that cannot do it is not usable here.

One deployment note that cost an hour: the API sits behind Cloudflare, which
rejects the default urllib user agent outright with a 403 and a Cloudflare error
code rather than an API error. The user agent below is not decoration.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# A small, fast, instruction following model that honours a JSON schema.
# Reading a planning brief is extraction, not reasoning, so the larger model
# buys accuracy this task does not need and latency a demo notices.
DEFAULT_MODEL = os.environ.get("PLANENG_GROQ_MODEL", "openai/gpt-oss-20b")

# Cloudflare blocks the stdlib default. See the module docstring.
USER_AGENT = "planeng/0.1 (spatial planning engine)"

# The free tier meters tokens per minute, not requests, and a planning brief
# carries the whole facility schema, so a handful of briefs in quick succession
# exhausts the minute. The allowance refills in about twenty seconds, which is
# worth waiting for and not worth failing a planner over.
MAX_RETRIES = 3
MAX_WAIT_S = 30.0


class GroqError(RuntimeError):
    """Groq could not answer. Callers fall back rather than failing the request."""


@dataclass
class Completion:
    text: str
    model: str
    duration_s: float
    eval_tokens: int
    ran_on_cpu: bool = False          # not meaningful remotely; kept for parity

    @property
    def tokens_per_second(self) -> float:
        return self.eval_tokens / self.duration_s if self.duration_s > 0 else 0.0


def api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def available(timeout: float = 4.0) -> bool:
    """Is a hosted model reachable with the key we hold?

    Checked rather than assumed, because the fallback to the rule based reader
    has to happen before a planner is left waiting on a request that cannot
    succeed.
    """
    if not api_key():
        return False
    try:
        _request("GET", "/models", timeout=timeout)
        return True
    except Exception:
        return False


def installed_models(timeout: float = 8.0) -> list[str]:
    try:
        data = _request("GET", "/models", timeout=timeout)
        return sorted(m["id"] for m in data.get("data", []))
    except Exception:
        return []


def chat(
    messages: list[dict],
    *,
    schema: dict | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    temperature: float = 0.0,
) -> Completion:
    """One completion, constrained to a JSON schema when one is given.

    Temperature defaults to 0 for the same reason as the local path: reading a
    brief is extraction, and sampling variety means the same words produce a
    different population each time it is run.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "planning_brief", "schema": schema},
        }

    started = time.perf_counter()
    data = _with_retry(lambda: _request("POST", "/chat/completions",
                                        body=body, timeout=timeout))

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise GroqError(f"Unexpected response shape: {str(data)[:200]}") from exc

    usage = data.get("usage") or {}
    return Completion(
        text=text,
        model=data.get("model", model),
        duration_s=time.perf_counter() - started,
        eval_tokens=int(usage.get("completion_tokens", 0)),
    )


def chat_json(messages: list[dict], schema: dict, **kwargs
              ) -> tuple[dict, Completion]:
    """Constrained completion, parsed. Raises GroqError if it is not JSON."""
    completion = chat(messages, schema=schema, **kwargs)
    try:
        return json.loads(completion.text), completion
    except json.JSONDecodeError as exc:
        raise GroqError(
            f"Model returned text that is not JSON despite a schema "
            f"constraint: {completion.text[:200]}"
        ) from exc


# ---------------------------------------------------------------------------

def _with_retry(call):
    """Run a request, waiting out a rate limit rather than failing on it.

    Only 429 is retried, and only while the wait is short. Every other status is
    a real error: retrying a bad key or a bad model name just makes the planner
    wait longer for the same answer.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise GroqError(
                    f"Groq returned HTTP {exc.code}: {detail}") from exc
            wait = _retry_after(exc)
            if wait > MAX_WAIT_S:
                raise GroqError(
                    f"Rate limited for {wait:.0f}s, which is longer than this "
                    f"is worth waiting. Falling back.") from exc
            time.sleep(wait)
        except Exception as exc:
            raise GroqError(f"Could not reach Groq: {exc}") from exc
    raise GroqError("Rate limited repeatedly.")


def _retry_after(exc: urllib.error.HTTPError) -> float:
    """How long the server asked us to wait, in seconds.

    Prefers the header. Falls back to the token reset hint, which comes as a
    duration string like `19.117s` or `14m24s`, and then to a fixed pause.
    """
    header = exc.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_WAIT_S)
        except ValueError:
            pass

    reset = exc.headers.get("x-ratelimit-reset-tokens") or ""
    seconds = 0.0
    number = ""
    for char in reset:
        if char.isdigit() or char == ".":
            number += char
        elif char == "m":
            seconds += float(number or 0) * 60
            number = ""
        elif char == "s":
            seconds += float(number or 0)
            number = ""
    return min(seconds, MAX_WAIT_S) if seconds else 5.0


def _request(method: str, path: str, body: dict | None = None,
             timeout: float = 30.0) -> dict:
    key = api_key()
    if not key:
        raise GroqError(
            "No GROQ_API_KEY is set. Put it in a .env file beside the project "
            "or in the host's secret store; it must never be committed.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=payload, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
