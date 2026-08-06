"""Model endpoint, token accounting, and nothing else.

Where the model runs is a config change, never a code change. Three profiles:

    ARMLLM_PROFILE=nvidia   https://inference-api.nvidia.com   (needs a key)
    ARMLLM_PROFILE=vllm     http://localhost:8000/v1           (class GPU)
    ARMLLM_PROFILE=mock     no network at all

Put your key in `shared/.env`, which is gitignored:

    ARMLLM_PROFILE=nvidia
    ARMLLM_API_KEY=nv-...
    ARMLLM_MODEL=azure/openai/gpt-5.5

**Never commit a key.** This directory is published. If you are on NVIDIA
infrastructure, that endpoint is internal — it is for building and rehearsing,
not for handing to a room.

Check your setup before you trust anything downstream:

    uv run python llm.py check

Token accounting lives here rather than in the agent because cost is a
first-class result, not an afterthought: the leaderboard ranks on the
accuracy/token Pareto front, so an equally accurate agent that spends fewer
tokens wins.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROFILES = {
    "nvidia": {"base_url": "https://inference-api.nvidia.com",
               "model": "azure/openai/gpt-5.5"},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3-8B"},
    "mock": {"base_url": "", "model": "mock"},
}


def load_env(path: Path | None = None) -> None:
    """Minimal .env loader — no dependency, no surprises. Existing vars win."""
    env_path = path or (HERE / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env()


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "retries": self.retries,
        }


class LLMError(RuntimeError):
    pass


class LLM:
    """Thin wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        profile: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
    ) -> None:
        self.profile = profile or os.environ.get("ARMLLM_PROFILE", "mock")
        if self.profile not in PROFILES:
            raise LLMError(f"unknown profile {self.profile!r}; expected one of {list(PROFILES)}")
        defaults = PROFILES[self.profile]

        self.model = model or os.environ.get("ARMLLM_MODEL") or defaults["model"]
        self.base_url = base_url or os.environ.get("ARMLLM_BASE_URL") or defaults["base_url"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.usage = Usage()
        self.mock = self.profile == "mock"
        # Qwen3.5/3.6 think before every tool call -- hundreds of tokens to emit
        # one `search`. For an agent the tool call IS the action, so thinking is
        # mostly latency. Off by default; ARMLLM_THINKING=1 to keep it.
        self.thinking = os.environ.get("ARMLLM_THINKING", "0") == "1"

        # Some newer models reject temperature != 1 or require
        # max_completion_tokens. We discover that once, then stop asking.
        self._send_temperature = True
        self._token_param = "max_tokens"

        if self.mock:
            self._client = None
            return

        key = (
            api_key
            or os.environ.get("ARMLLM_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise LLMError(
                f"profile {self.profile!r} needs an API key. Put ARMLLM_API_KEY in "
                f"{HERE / '.env'} (gitignored), or export it. Use ARMLLM_PROFILE=mock "
                f"to run offline."
            )
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=120.0, max_retries=0)

    # ----------------------------------------------------------------
    def describe(self) -> str:
        return (f"profile={self.profile} model={self.model} base_url={self.base_url or '-'} thinking={self.thinking}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the assistant message dict. Records usage as a side effect."""
        if self.mock:
            return _mock_chat(messages, self.usage)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._once(messages, tools)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                text = str(exc)
                if self._adapt(text):
                    continue  # parameter problem, fixed — retry immediately
                if attempt == self.max_retries or not _is_transient(text):
                    break
                self.usage.retries += 1
                time.sleep(min(2 ** attempt, 8) + random.random())
        raise LLMError(f"{self.describe()} failed after retries: {last_error}") from last_error

    def _once(self, messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if self._send_temperature:
            kwargs["temperature"] = self.temperature
        kwargs[self._token_param] = self.max_tokens
        if not self.thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        resp = self._client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ]
            or None,
        }

    def _adapt(self, error_text: str) -> bool:
        """Learn the endpoint's dialect from its complaints. Returns True if we changed
        something and the call is worth retrying immediately."""
        low = error_text.lower()
        if "max_completion_tokens" in low and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            return True
        if "temperature" in low and self._send_temperature and (
            "unsupported" in low or "does not support" in low or "only the default" in low
        ):
            self._send_temperature = False
            return True
        return False


def _is_transient(error_text: str) -> bool:
    low = error_text.lower()
    return any(
        marker in low
        for marker in (
            "rate limit", "429", "timeout", "timed out", "502", "503", "504",
            "overloaded", "connection", "temporarily",
        )
    )


# --------------------------------------------------------------------------
# Mock backend — deterministic, offline, deliberately naive.
#
# It does exactly what an unengineered agent does: search once, read the single
# highest-ranked document, answer with the first figure it sees. On most tasks
# that works. On a task whose top-ranked document has been superseded by a later
# notice it is confidently wrong — which is the 09:05 cold open, reproducible on
# a laptop with no network.
# --------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mock_chat(messages: list[dict], usage: Usage) -> dict:
    prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
    tool_results = [m for m in messages if m.get("role") == "tool"]

    if not tool_results:
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = str(m.get("content") or "")
        out = _call("call_1", "search", {"query": query, "k": 5})
    elif len(tool_results) == 1:
        first = str(tool_results[0].get("content") or "")
        m = re.search(r'"id":\s*"([^"]+)"', first)
        out = _call("call_2", "read_doc", {"doc_id": m.group(1) if m else "office-central"})
    else:
        body = str(tool_results[-1].get("content") or "")
        nums = re.findall(r"(\d[\d,]*)\s*AMD", body)
        answer = nums[0] + " AMD" if nums else "Unable to determine from the documents."
        out = _call("call_3", "finish", {"answer": answer})

    usage.add(_approx_tokens(" " * prompt_chars), _approx_tokens(json.dumps(out)))
    return out


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


# --------------------------------------------------------------------------


def check() -> int:
    """Verify the endpoint answers, and — the part that actually breaks — that it
    returns tool calls in the shape the agent loop expects."""
    llm = LLM()
    print(f"config  : {llm.describe()}")

    t0 = time.time()
    plain = llm.chat([{"role": "user", "content": "Reply with exactly: ok"}])
    print(f"chat    : {(plain.get('content') or '').strip()[:40]!r}  ({time.time() - t0:.1f}s)")

    from tools import TOOL_SCHEMAS

    t0 = time.time()
    msg = llm.chat(
        [
            {"role": "system", "content": "Use the tools. Search before answering."},
            {"role": "user", "content": "What is the fee for a zoning certificate in Dvin?"},
        ],
        tools=TOOL_SCHEMAS,
    )
    calls = msg.get("tool_calls")
    if not calls:
        print("tools   : NO TOOL CALL RETURNED — the agent loop will not work with this model")
        print(f"          content was: {(msg.get('content') or '')[:120]!r}")
        return 1
    fn = calls[0]["function"]
    print(f"tools   : {fn['name']}({fn['arguments'][:70]})  ({time.time() - t0:.1f}s)")
    print(f"usage   : {llm.usage.as_dict()}")
    print("\nOK — endpoint answers and speaks tool-calls.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(check() if "check" in sys.argv[1:] else print(LLM().describe()))
