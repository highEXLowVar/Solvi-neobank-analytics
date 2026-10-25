"""OpenAI Responses API client with a request-hash response cache.

The API offers no reliable determinism switch, so determinism comes from the
cache: each request is hashed (model + every parameter, canonical JSON) and
the response is stored under evals/cache/. Tool outputs are deterministic
DuckDB reads, so a cached first turn reproduces the second request byte for
byte, and the whole run replays offline with zero API calls. The latency
recorded on the first live call is kept with the response, so replayed
benchmarks report real wall-clock numbers rather than disk reads.

The workflows speak in simple content-block messages (text / tool_use /
tool_result). This client translates them to the Responses API and the reply
back. Reasoning items are not echoed back between turns: the loop is
stateless (store=False) and the tool results carry the state.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "evals" / "cache"


class CacheMiss(RuntimeError):
    pass


def _load_dotenv() -> None:
    """API keys can live in a gitignored .env at the repo root; it wins over the shell."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not line.lstrip().startswith("#"):
                os.environ[k.strip()] = v.strip().strip('"')


MODEL = "gpt-6-luna"

# USD per million tokens: input, cached input, output.
# from developers.openai.com/api/docs/pricing, 25 Oct 2026, short context
PRICES: dict[str, dict[str, float]] = {
    "gpt-6-luna": {"in": 0.10, "cache_read": 0.01, "out": 0.50},
    "gpt-5.6-luna": {"in": 0.20, "cache_read": 0.02, "out": 1.20},
    "gpt-5.4-nano": {"in": 0.20, "cache_read": 0.02, "out": 1.25},
    "gpt-5.4-mini": {"in": 0.75, "cache_read": 0.075, "out": 4.50},
    "gpt-6.1-sol": {"in": 2.00, "cache_read": 0.10, "out": 10.00},
    "gpt-6-sol": {"in": 2.00, "cache_read": 0.20, "out": 10.00},
}


def cost_usd(usage: dict, model: str) -> float:
    p = PRICES.get(model)
    if not p:
        return 0.0
    cached = usage.get("cache_read_input_tokens", 0)
    return ((usage.get("input_tokens", 0) - cached) * p["in"] + cached * p["cache_read"]
            + usage.get("output_tokens", 0) * p["out"]) / 1e6


def _to_input(messages: list) -> list:
    items = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            items.append({"role": m["role"], "content": content})
            continue
        for b in content:
            t = b["type"]
            if t == "text" and b.get("text"):
                items.append({"role": m["role"], "content": b["text"]})
            elif t == "tool_use":
                items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                              "arguments": json.dumps(b["input"])})
            elif t == "tool_result":
                items.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                              "output": b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])})
    return items


def _to_tools(tools: list) -> list:
    return [{"type": "function", "name": t["name"], "description": t["description"],
             "parameters": t["input_schema"], "strict": True} for t in tools]


class LLM:
    def __init__(self, model: str = MODEL, effort: str | None = None, offline: bool = False,
                 cache_dir: Path = CACHE_DIR):
        self.model, self.offline, self.cache_dir = model, offline, cache_dir
        self.effort = effort or os.environ.get("SOLVI_EFFORT", "low")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            _load_dotenv()
            self._client = openai.OpenAI()
        return self._client

    def cost(self, usage: dict) -> float:
        return cost_usd(usage, self.model)

    def create(self, *, system: str, messages: list, json_schema: dict,
               tools: list | None = None, max_tokens: int = 16000) -> dict:
        params = {
            "model": self.model,
            "instructions": system,
            "input": _to_input(messages),
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": self.effort},
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": json_schema,
                                "strict": True}},
            "store": False,
        }
        if tools:
            params["tools"] = _to_tools(tools)
        key = "oai-" + hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            return {**rec["response"], "_latency_s": rec["latency_s"], "_cached": True, "_key": key}
        if self.offline:
            raise CacheMiss(key)

        t0 = time.perf_counter()
        resp = self.client.responses.create(**params)
        latency = time.perf_counter() - t0
        content = []
        for item in resp.output:
            if item.type == "message":
                text = "".join(c.text for c in item.content if c.type == "output_text")
                if text:
                    content.append({"type": "text", "text": text})
            elif item.type == "function_call":
                content.append({"type": "tool_use", "id": item.call_id, "name": item.name,
                                "input": json.loads(item.arguments)})
        u = resp.usage
        out = {
            "model": resp.model,
            "content": content,
            "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in content)
            else ("max_tokens" if resp.status == "incomplete" else "end_turn"),
            "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                      "cache_read_input_tokens": (u.input_tokens_details.cached_tokens or 0)
                      if u.input_tokens_details else 0,
                      "cache_creation_input_tokens": 0},
        }
        path.write_text(json.dumps({"latency_s": round(latency, 3), "response": out}, indent=1))
        return {**out, "_latency_s": round(latency, 3), "_cached": False, "_key": key}


def text_of(resp: dict) -> str:
    return "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")


def assistant_turn(resp: dict) -> dict:
    """the response's content blocks, echoed back unchanged."""
    return {"role": "assistant", "content": resp["content"]}


def make_llm(offline: bool = False) -> LLM:
    """gpt-6-luna by default: it is what the committed results were recorded
    with (evals/results/eval_results_trial_*.json has the comparison that
    picked it). SOLVI_MODEL overrides it."""
    return LLM(model=os.environ.get("SOLVI_MODEL") or MODEL, offline=offline)
