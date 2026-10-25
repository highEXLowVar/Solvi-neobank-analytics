"""The three triage workflows the evals compare.

A  raw_text_to_sql   one SQL query against the raw event tables, given only
                     table and column names, then a verdict. no retry.
B  dbt_single_pass   one SQL query against the dbt marts, given the documented
                     semantic layer, then a verdict. no retry.
C  tool_agent        multi-step tool loop over the marts: schema lookup, SQL
                     with the DuckDB error fed back for self-correction, the
                     stats tool, and the unit-economics guardrail (in the
                     prompt, and enforced in code before a SHIP is accepted).

Every workflow returns a RunRecord: the verdict plus everything needed to
grade it (each query, each tool output) and to render the trace on the site.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from pydantic import ValidationError

from . import prompts as P
from .llm import LLM, assistant_turn, text_of
from .schema import ProductTriageVerdict, SqlDraft, strict_json_schema
from .stats import calculate_uplift_or_stats
from .warehouse import Warehouse

VERDICT_SCHEMA = strict_json_schema(ProductTriageVerdict)
SQL_SCHEMA = strict_json_schema(SqlDraft)
MAX_TURNS = 14

CONFIGS = {
    "A_raw_text_to_sql": "Raw text-to-SQL",
    "B_dbt_single_pass": "Single-pass dbt SQL",
    "C_tool_agent": "Multi-step agent + guardrails",
}
# same agent with the unit-economics guardrail switched off, to check whether
# the guardrail is what actually stops it shipping the bonus
ABLATIONS = {"C_no_guardrail": "Agent without the guardrail"}

TOOLS = [
    {
        "name": "get_dbt_schema_context",
        "description": "Documented dbt semantic layer: model descriptions, column docs and metric "
                       "definitions. Pass a model name for one model, or null for all of them.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"mart_name": {"type": ["string", "null"]}},
            "required": ["mart_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "execute_readonly_duckdb_sql",
        "description": "Run one read-only DuckDB SELECT/WITH query against the dbt marts. Returns at "
                       "most 200 rows. On a syntax or binder error you get DuckDB's exact message back.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate_uplift_or_stats",
        "description": (
            "Vetted estimators computed live from the marts. analysis = "
            "'kyc_changepoint' (platform, doc_type: binomial LLR changepoint scan of daily KYC "
            "approvals, break/recovery dates, levels, closest app release); "
            "'channel_unit_economics' (channel or null: CAC per activated user, 36-month LTV, "
            "payback month, implied monthly churn); "
            "'ab_test_readout' (segment_by: channel/country/device/age_band or null: TOPUP10 "
            "conversion lift, z-test, bonus cost per incremental conversion, 60-day margin "
            "delta net of bonus); "
            "'two_proportion_test' (k_a, n_a, k_b, n_b). Unused fields are null."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_slice": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": [
                            "kyc_changepoint", "channel_unit_economics",
                            "ab_test_readout", "two_proportion_test"]},
                        "platform": {"type": ["string", "null"]},
                        "doc_type": {"type": ["string", "null"]},
                        "channel": {"type": ["string", "null"]},
                        "segment_by": {"type": ["string", "null"]},
                        "k_a": {"type": ["number", "null"]},
                        "n_a": {"type": ["number", "null"]},
                        "k_b": {"type": ["number", "null"]},
                        "n_b": {"type": ["number", "null"]},
                    },
                    "required": ["analysis", "platform", "doc_type", "channel", "segment_by",
                                 "k_a", "n_a", "k_b", "n_b"],
                    "additionalProperties": False,
                }
            },
            "required": ["metric_slice"],
            "additionalProperties": False,
        },
    },
]


def _py(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def dump(o) -> str:
    return json.dumps(o, default=_py)


@dataclass
class RunRecord:
    config: str
    case_id: str
    question: str
    verdict: dict | None = None
    parse_error: str | None = None
    steps: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0,
                                                  "cache_creation_input_tokens": 0,
                                                  "cache_read_input_tokens": 0})
    llm_calls: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    guardrail_rejections: int = 0
    stop_reason: str | None = None

    def add_usage(self, resp: dict, llm):
        u = resp.get("usage") or {}
        for k in self.usage:
            self.usage[k] += u.get(k) or 0
        self.llm_calls += 1
        self.latency_s += resp["_latency_s"]
        self.cost_usd = round(llm.cost(self.usage), 5)
        self.stop_reason = resp.get("stop_reason")

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_verdict(rec: RunRecord, resp: dict) -> None:
    raw = text_of(resp)
    try:
        v = ProductTriageVerdict.model_validate_json(raw)
        rec.verdict = v.model_dump()
        rec.steps.append({"kind": "verdict", "verdict": rec.verdict})
    except ValidationError as e:
        rec.parse_error = f"{e.error_count()} validation errors; raw={raw[:300]!r}"
        rec.steps.append({"kind": "verdict_error", "text": rec.parse_error})


def _run_sql(rec: RunRecord, wh: Warehouse, sql: str) -> dict:
    res = wh.run(sql)
    rec.queries.append({"sql": res.sql, "ok": res.ok, "error": res.error,
                        "rows": len(res.rows)})
    payload = res.to_tool_payload()
    rec.tool_outputs.append(dump(payload))
    return payload


def _question_text(question: str, as_of: str | None) -> str:
    return f"(Today is {as_of}.) {question}" if as_of else question


# ---- A and B ---------------------------------------------------------------

def single_pass(llm: LLM, case_id: str, question: str, source: str,
                as_of: str | None = None) -> RunRecord:
    config = "A_raw_text_to_sql" if source == "raw" else "B_dbt_single_pass"
    rec = RunRecord(config, case_id, question)
    wh = Warehouse(source, as_of)
    schema = wh.bare_schema() if source == "raw" else wh.dbt_context()
    q = _question_text(question, as_of)
    rec.steps.append({"kind": "user_prompt", "text": q})
    system = P.ANALYST_BRIEF + "\n" + P.SINGLE_PASS_SQL
    msgs = [{"role": "user", "content": f"Warehouse schema:\n{schema}\n\nQuestion: {q}"}]
    r1 = llm.create(system=system, messages=msgs, json_schema=SQL_SCHEMA, max_tokens=8000)
    rec.add_usage(r1, llm)
    try:
        draft = SqlDraft.model_validate_json(text_of(r1))
    except ValidationError:
        rec.parse_error = "no SQL draft returned"
        return rec
    rec.steps.append({"kind": "tool_call", "tool": "execute_readonly_duckdb_sql",
                      "input": {"query": draft.sql}, "note": draft.rationale})
    payload = _run_sql(rec, wh, draft.sql)
    rec.steps.append({"kind": "tool_result", "tool": "execute_readonly_duckdb_sql",
                      "ok": payload["ok"], "output": payload})
    msgs += [assistant_turn(r1),
             {"role": "user", "content": f"{P.SINGLE_PASS_VERDICT}\n\nQuery result:\n{dump(payload)}"}]
    r2 = llm.create(system=system, messages=msgs, json_schema=VERDICT_SCHEMA)
    rec.add_usage(r2, llm)
    _parse_verdict(rec, r2)
    return rec


# ---- C ---------------------------------------------------------------------

_ECON_SQL = re.compile(r"margin|ltv|cac|bonus_cost|cost", re.I)


def _checked_economics(rec: RunRecord) -> bool:
    for s in rec.steps:
        if s["kind"] != "tool_call":
            continue
        if s["tool"] == "calculate_uplift_or_stats" and s["input"]["metric_slice"]["analysis"] in (
                "ab_test_readout", "channel_unit_economics"):
            return True
        if s["tool"] == "execute_readonly_duckdb_sql" and _ECON_SQL.search(s["input"]["query"]):
            return True
    return False


def _exec_tool(rec: RunRecord, wh: Warehouse, name: str, args: dict) -> tuple[dict | str, bool]:
    if name == "execute_readonly_duckdb_sql":
        payload = _run_sql(rec, wh, args["query"])
        return payload, payload["ok"]
    if name == "get_dbt_schema_context":
        return wh.dbt_context(args.get("mart_name")), True
    if name == "calculate_uplift_or_stats":
        out = calculate_uplift_or_stats(wh, args["metric_slice"])
        rec.tool_outputs.append(dump(out))
        return out, "error" not in out
    return {"error": f"unknown tool {name}"}, False


def tool_agent(llm: LLM, case_id: str, question: str, as_of: str | None = None,
               guardrail: bool = True) -> RunRecord:
    rec = RunRecord("C_tool_agent", case_id, question)
    wh = Warehouse("marts", as_of)
    q = _question_text(question, as_of)
    rec.steps.append({"kind": "user_prompt", "text": q})
    system = "\n".join([P.ANALYST_BRIEF, P.TOOL_AGENT] + ([P.UNIT_ECONOMICS_GUARDRAIL] if guardrail else []))
    msgs: list = [{"role": "user", "content": q}]
    for turn in range(MAX_TURNS + 1):
        resp = llm.create(system=system, messages=msgs, json_schema=VERDICT_SCHEMA, tools=TOOLS)
        rec.add_usage(resp, llm)
        msgs.append(assistant_turn(resp))
        calls = [b for b in resp["content"] if b["type"] == "tool_use"]
        if resp.get("stop_reason") == "tool_use" and calls:
            for b in resp["content"]:
                if b["type"] == "text" and b["text"].strip():
                    rec.steps.append({"kind": "model_note", "text": b["text"].strip()})
            results = []
            for b in calls:
                rec.steps.append({"kind": "tool_call", "tool": b["name"], "input": b["input"]})
                out, ok = _exec_tool(rec, wh, b["name"], b["input"])
                rec.steps.append({"kind": "tool_result", "tool": b["name"], "ok": ok, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b["id"],
                                "content": out if isinstance(out, str) else dump(out),
                                **({} if ok else {"is_error": True})})
            if turn == MAX_TURNS - 1:
                results.append({"type": "text", "text": P.TOOL_BUDGET_EXHAUSTED})
            msgs.append({"role": "user", "content": results})
            continue
        if resp.get("stop_reason") != "end_turn":
            rec.parse_error = f"stopped: {resp.get('stop_reason')}"
            return rec
        _parse_verdict(rec, resp)
        if (guardrail and rec.verdict and rec.verdict["recommended_action"] == "SHIP"
                and not _checked_economics(rec) and rec.guardrail_rejections == 0):
            rec.guardrail_rejections += 1
            rec.steps.append({"kind": "guardrail", "text": "SHIP rejected: unit economics not checked"})
            rec.verdict = None
            msgs.append({"role": "user", "content": P.GUARDRAIL_REJECTION.format(action="SHIP")})
            continue
        return rec
    rec.parse_error = rec.parse_error or "turn limit reached"
    return rec


def run(llm: LLM, config: str, case_id: str, question: str, as_of: str | None = None) -> RunRecord:
    """latency_s on the record is model wall-clock as recorded on the live
    call (DuckDB time is excluded so replays give identical numbers; it is
    well under a second per triage)."""
    if config == "A_raw_text_to_sql":
        return single_pass(llm, case_id, question, "raw", as_of)
    if config == "B_dbt_single_pass":
        return single_pass(llm, case_id, question, "marts", as_of)
    if config == "C_no_guardrail":
        rec = tool_agent(llm, case_id, question, as_of, guardrail=False)
        rec.config = config
        return rec
    return tool_agent(llm, case_id, question, as_of)
