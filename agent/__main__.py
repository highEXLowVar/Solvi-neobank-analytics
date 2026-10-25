"""Ask the triage agent one question.

    python -m agent "Why did KYC conversion drop in recent weeks?" [--as-of 2025-12-15]
"""

from __future__ import annotations

import argparse
import json

from .llm import make_llm
from .workflows import CONFIGS, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--as-of")
    ap.add_argument("--config", default="C_tool_agent", choices=list(CONFIGS))
    args = ap.parse_args()
    rec = run(make_llm(), args.config, "adhoc", args.question, args.as_of)
    for s in rec.steps:
        if s["kind"] == "tool_call":
            print(f"> {s['tool']}: {json.dumps(s['input'])[:300]}")
        elif s["kind"] == "tool_result":
            print(f"  {'ok' if s['ok'] else 'ERROR'}: {json.dumps(s['output'], default=str)[:300]}")
    print(json.dumps(rec.verdict, indent=1) if rec.verdict else rec.parse_error)
    print(f"\n{rec.llm_calls} model calls, {rec.latency_s:.1f}s, ${rec.cost_usd:.3f}")


if __name__ == "__main__":
    main()
