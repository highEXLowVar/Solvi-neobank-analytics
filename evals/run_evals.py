"""Benchmark the three triage workflows against the planted ground truth.

    python -m evals.run_evals                 # live where the cache misses, needs OPENAI_API_KEY
    python -m evals.run_evals --offline       # replay evals/cache only, zero API calls
    python -m evals.run_evals --site ..\\parkhouse.fr\\src\\data\\solvi_agent.json

Writes evals/results/eval_results.json (scores) and agent_traces.json (every
step of every run). Re-running is free and gives identical files: responses
are cached by request hash (see agent/llm.py) and DuckDB is deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent.llm import CacheMiss, make_llm
from agent.workflows import ABLATIONS, CONFIGS, run

from .cases import CASES, N_CHECKS
from .grading import dumps, grade_run, summarise

OUT = Path(__file__).resolve().parent / "results"
SITE_ROWS = 8


def _trim(step: dict) -> dict:
    s = dict(step)
    out = s.get("output")
    if isinstance(out, str):
        s["output"] = out if len(out) <= 700 else out[:700] + " …"
    elif isinstance(out, dict) and "rows" in out:
        s["output"] = {**out, "rows": out["rows"][:SITE_ROWS],
                       "rows_shown": min(SITE_ROWS, len(out["rows"]))}
    return s


def site_payload(summary: dict, grades: list[dict], records: list[dict]) -> dict:
    by = {(r["config"], r["case_id"]): r for r in records}
    gby = {(g["config"], g["case_id"]): g for g in grades}
    traces = []
    for c in CASES:
        if not c.get("site"):
            continue
        r = by.get(("C_tool_agent", c["id"]))
        if r:
            traces.append({"case_id": c["id"], "question": c["question"], "as_of": c["as_of"],
                           "grade": gby[("C_tool_agent", c["id"])],
                           "steps": [_trim(s) for s in r["steps"]]})
    # the instructive failure: what A and B said about the bonus, next to C
    contrast = {cfg: {"verdict": by[(cfg, "topup10_rollout")]["verdict"],
                      "grade": gby[(cfg, "topup10_rollout")],
                      "queries": by[(cfg, "topup10_rollout")]["queries"]}
                for cfg in list(CONFIGS) + list(ABLATIONS) if (cfg, "topup10_rollout") in by}
    return {"configs": CONFIGS, "ablations": ABLATIONS, "summary": summary, "n_checks": N_CHECKS,
            "matrix": [{"case_id": g["case_id"], "config": g["config"], "decision_ok": g["decision_ok"],
                        "action": g["action"], "expected_action": g["expected_action"],
                        "hits": sum(t["hit"] for t in g["targets"]), "n_targets": len(g["targets"]),
                        "false_positive": g["false_positive"], "fabricated": len(g["fabricated"])}
                       for g in grades],
            "bonus_contrast": contrast, "traces": traces}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="replay the cache, never call the API")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS) + list(ABLATIONS))
    ap.add_argument("--cases", nargs="*", default=[c["id"] for c in CASES])
    ap.add_argument("--site", type=Path, help="also write the dashboard JSON here")
    ap.add_argument("--model", help="override SOLVI_MODEL for this run")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. a model trial")
    args = ap.parse_args()
    if args.model:
        os.environ["SOLVI_MODEL"] = args.model

    llm = make_llm(offline=args.offline)
    records, grades, missing = [], [], []
    for cfg in args.configs:
        for case in CASES:
            if case["id"] not in args.cases:
                continue
            try:
                rec = run(llm, cfg, case["id"], case["question"], case["as_of"]).to_dict()
            except CacheMiss:
                missing.append(f"{cfg}/{case['id']}")
                continue
            g = grade_run(rec, case)
            records.append(rec)
            grades.append(g)
            hits = sum(t["hit"] for t in g["targets"])
            print(f"{cfg:<20} {case['id']:<26} action={str(g['action']):<26} "
                  f"ok={g['decision_ok']!s:<5} hits={hits}/{len(g['targets'])} "
                  f"fab={len(g['fabricated'])} q={g['n_queries']}({g['n_failed_queries']} err) "
                  f"{g['latency_s']:.1f}s ${g['cost_usd']:.3f}")
    if missing:
        print(f"\n{len(missing)} runs not in cache (offline): {', '.join(missing)}")

    summary = {cfg: summarise([g for g in grades if g["config"] == cfg])
               for cfg in args.configs if any(g["config"] == cfg for g in grades)}
    OUT.mkdir(exist_ok=True)
    complete = not missing and len(args.cases) == len(CASES) and set(args.configs) >= set(CONFIGS)
    results = {
        "complete": complete,
        "model": llm.model, "effort": llm.effort,
        "n_cases": len(CASES), "n_ground_truth_checks": N_CHECKS,
        "n_noise_traps": sum(bool(c.get("noise")) for c in CASES),
        "configs": CONFIGS, "ablations": ABLATIONS, "summary": summary, "per_case": grades,
    }
    (OUT / f"eval_results{args.tag}.json").write_text(dumps(results))
    (OUT / f"agent_traces{args.tag}.json").write_text(dumps(records))
    print("\n" + json.dumps({k: {"retrieval": f"{v['retrieval']['hits']}/{v['retrieval']['total']}",
                                 "decisions": f"{v['decisions']['correct']}/{v['decisions']['total']}",
                                 "supported": f"{v['supported_decisions']['correct']}/{v['supported_decisions']['total']}",
                                 "noise_fp": f"{v['noise_false_positives']['count']}/{v['noise_false_positives']['total']}",
                                 "fabricated": f"{v['fabricated']['count']}/{v['fabricated']['reported_values']}",
                                 "sql_valid": v["sql"]["valid_execution_rate"],
                                 "p50_s": v["latency_p50_s"], "usd": v["cost_per_triage_usd"]}
                             for k, v in summary.items()}, indent=1))
    if args.site:
        if not complete:
            raise SystemExit("not writing site JSON from an incomplete run")
        args.site.write_text(dumps({**site_payload(summary, grades, records), "model": llm.model}))
        print(f"wrote {args.site}")


if __name__ == "__main__":
    main()
