"""Deterministic grading. No LLM judge anywhere: every score is a comparison
between the verdict JSON and either a planted target or the tool outputs of
that same run."""

from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np

from .cases import TOL_REL

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
DATE_LONG = re.compile(r"(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})", re.I)


def _nums(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _dates(text: str) -> list[dt.date]:
    out = [dt.date.fromisoformat(d) for d in DATE.findall(text)]
    out += [dt.date(int(y), _MONTHS[mo.lower()], int(d)) for d, mo, y in DATE_LONG.findall(text)]
    return out


def verdict_text(v: dict) -> str:
    return " ".join([v["root_cause_hypothesis"], v["reasoning_summary"]] +
                    [m["name"] for m in v["supporting_metrics"]])


def verdict_numbers(v: dict) -> list[float]:
    vals = []
    for m in v["supporting_metrics"]:
        vals += [x for x in (m["baseline_value"], m["anomaly_value"]) if x is not None]
        # a metric reported as baseline -> anomaly states its change too
        # (control 0.6923 -> treatment 0.7488 is a 5.65pp lift)
        if m["baseline_value"] is not None and m["anomaly_value"] is not None:
            vals.append(m["anomaly_value"] - m["baseline_value"])
    # numbers quoted in prose count too, minus anything that is part of a date
    prose = DATE.sub(" ", " ".join([v["root_cause_hypothesis"], v["reasoning_summary"]]))
    return vals + _nums(prose)


def _close(reported: float, target: float, abs_tol: float = 0.0) -> bool:
    # a rate can come back as a fraction or as a percentage: accept either scale
    for x in (reported, reported / 100, reported * 100):
        if abs(x - target) <= max(TOL_REL * abs(target), abs_tol):
            return True
    return False


def hit(target: dict, v: dict | None, rec: dict | None = None) -> bool:
    """a target counts as retrieved only if the verdict states it AND (for
    numbers and dates) the value came back from a query or tool in that run,
    so a figure recalled from the question or guessed doesn't score."""
    if v is None:
        return False
    if target["kind"] == "category":
        t = verdict_text(v).lower().replace("_", " ")
        return all(w in t for w in target["all_of"])
    outputs = rec["tool_outputs"] if rec else None
    if target["kind"] == "date":
        want = dt.date.fromisoformat(target["value"])
        stated = [d for d in _dates(verdict_text(v)) if abs((d - want).days) <= 1]
        if outputs is None:
            return bool(stated)
        seen = {dt.date.fromisoformat(d) for o in outputs for d in DATE.findall(o)}
        return any(d in seen for d in stated)
    obs = _observed(outputs) if outputs is not None else None
    return any(_close(x, target["value"], target.get("abs_tol", 0.0)) and (obs is None or grounded(x, obs))
               for x in verdict_numbers(v))


# ---- grounding: did each reported number come back from DuckDB? ----------

def _observed(tool_outputs: list[str]) -> list[np.ndarray]:
    return [np.array(_nums(DATE.sub(" ", o))[:600], dtype=float) for o in tool_outputs]


def _decimals(x: float) -> int:
    s = repr(float(x))
    if "e" in s:
        return 6
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def grounded(value: float, observed: list[np.ndarray]) -> bool:
    """a reported number is grounded if, within 1% or its own rounding, it is
    an observed value, a percentage/fraction rescale of one, or the difference
    or ratio of two values from the same tool output (deltas and CAC/LTV
    style ratios are fine; inventing a figure is not)."""
    tol = lambda x: np.maximum(0.01 * np.abs(x), 0.5 * 10 ** (-_decimals(value)))  # noqa: E731
    for arr in observed:
        if arr.size == 0:
            continue
        for cand in (arr, arr * 100, arr / 100):
            if np.any(np.abs(cand - value) <= tol(cand)):
                return True
        a = arr[:150]
        diff = a[:, None] - a[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = a[:, None] / a[None, :]
        for cand in (diff, diff * 100, ratio, ratio * 100):
            c = cand[np.isfinite(cand)]
            if np.any(np.abs(c - value) <= tol(c)):
                return True
    return False


def fabricated_metrics(rec: dict) -> list[dict]:
    v = rec["verdict"]
    if v is None:
        return []
    obs = _observed(rec["tool_outputs"])
    bad = []
    for m in v["supporting_metrics"]:
        for field in ("baseline_value", "anomaly_value"):
            x = m[field]
            if x is not None and not grounded(x, obs):
                bad.append({"metric": m["name"], "field": field, "value": x})
    return bad


def grade_run(rec: dict, case: dict) -> dict:
    v = rec["verdict"]
    targets = [{"check": t["check"], "label": t["label"], "hit": hit(t, v, rec)} for t in case["targets"]]
    extra = [{"label": t["label"], "hit": hit(t, v, rec)} for t in case.get("extra_targets", [])]
    action = v["recommended_action"] if v else None
    decision_ok = None if case["expected_action"] is None else action == case["expected_action"]
    false_positive = None
    if case.get("noise"):
        false_positive = v is None or v["incident_detected"] or action != "NO_ACTION_NOISE"
    fab = fabricated_metrics(rec)
    # right call for the right reason: the decision is correct AND the run
    # actually retrieved the evidence for it (a planted number, or for a noise
    # trap at least one query that returned data). "don't ship, my query came
    # back empty" is a correct label with nothing behind it.
    evidence = any(t["hit"] for t in targets + extra) or (
        not targets and not extra and any(x["ok"] and x["rows"] > 0 for x in rec["queries"]))
    if not targets and not extra and case.get("noise"):
        evidence = evidence or any(s["kind"] == "tool_result" and s["ok"] and s["tool"] == "calculate_uplift_or_stats"
                                   for s in rec["steps"])
    decision_supported = None if decision_ok is None else bool(decision_ok and evidence)
    n_reported = sum(
        x is not None for m in (v["supporting_metrics"] if v else [])
        for x in (m["baseline_value"], m["anomaly_value"]))
    q = rec["queries"]
    failed = [i for i, x in enumerate(q) if not x["ok"]]
    recovered = sum(any(y["ok"] for y in q[i + 1:]) for i in failed)
    return {
        "case_id": case["id"], "config": rec["config"],
        "action": action, "expected_action": case["expected_action"], "decision_ok": decision_ok,
        "decision_supported": decision_supported,
        "incident_detected": v["incident_detected"] if v else None,
        "false_positive": false_positive,
        "targets": targets, "extra_targets": extra,
        "fabricated": fab, "n_reported_values": n_reported,
        "n_queries": len(q), "n_failed_queries": len(failed), "n_recovered": recovered,
        "parse_error": rec["parse_error"],
        "latency_s": round(rec["latency_s"], 2), "cost_usd": rec["cost_usd"],
        "llm_calls": rec["llm_calls"], "usage": rec["usage"],
        "guardrail_rejections": rec["guardrail_rejections"],
    }


def summarise(grades: list[dict]) -> dict:
    total_checks = sum(len(g["targets"]) for g in grades)
    hits = sum(t["hit"] for g in grades for t in g["targets"])
    dec = [g["decision_ok"] for g in grades if g["decision_ok"] is not None]
    sup = [g["decision_supported"] for g in grades if g["decision_supported"] is not None]
    noise = [g["false_positive"] for g in grades if g["false_positive"] is not None]
    rep = sum(g["n_reported_values"] for g in grades)
    fab = sum(len(g["fabricated"]) for g in grades)
    halluc_runs = sum(bool(g["fabricated"]) or bool(g["false_positive"]) for g in grades)
    nq = sum(g["n_queries"] for g in grades)
    nfail = sum(g["n_failed_queries"] for g in grades)
    nrec = sum(g["n_recovered"] for g in grades)
    lat = [g["latency_s"] for g in grades]
    cost = [g["cost_usd"] for g in grades]
    return {
        "retrieval": {"hits": hits, "total": total_checks},
        "decision_accuracy": round(sum(dec) / len(dec), 4) if dec else None,
        "decisions": {"correct": sum(dec), "total": len(dec)},
        "supported_decisions": {"correct": sum(sup), "total": len(sup)},
        "false_positive_rate_noise": round(sum(noise) / len(noise), 4) if noise else None,
        "noise_false_positives": {"count": sum(noise), "total": len(noise)},
        "fabricated_metric_rate": round(fab / rep, 4) if rep else 0.0,
        "fabricated": {"count": fab, "reported_values": rep},
        "hallucination_rate": round(halluc_runs / len(grades), 4),
        "sql": {"queries": nq, "failed": nfail, "recovered": nrec,
                "valid_execution_rate": round((nq - nfail) / nq, 4) if nq else None,
                "self_correction_rate": round(nrec / nfail, 4) if nfail else None},
        "latency_p50_s": round(float(np.median(lat)), 2),
        "cost_per_triage_usd": round(float(np.mean(cost)), 4),
        "cost_total_usd": round(float(np.sum(cost)), 4),
        "parse_failures": sum(g["parse_error"] is not None for g in grades),
    }


def dumps(o) -> str:
    return json.dumps(o, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))

