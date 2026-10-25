"""Read-only access to the warehouse for the triage agent.

Two databases can sit behind the agent:

* ``marts``: the dbt build in data/solvi.duckdb (marts + exports only, the
  staging views point at parquet paths and are not exposed).
* ``raw``: the simulator's raw event files loaded as plain tables into
  data/raw_events.duckdb, no dbt anywhere. this is what baseline A gets.

Every session is an in-memory DuckDB that ATTACHes the file READ_ONLY and
exposes each table as a view, optionally cut at an ``as_of`` date so a case
can be asked "as of mid-December" without the agent peeking at the future.
After the views exist, file system access is switched off, so nothing can
read_parquet() its way into data/ground_truth/.
"""

from __future__ import annotations

import datetime as dt
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
MARTS_DB = ROOT / "data" / "solvi.duckdb"
RAW_DB = ROOT / "data" / "raw_events.duckdb"
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "warehouse" / "models"

ROW_LIMIT = 200
TIMEOUT_S = 20.0

# date column used to cut each table at as_of. tables not listed are
# dimension-like (channels, releases) or already keyed by a month that the
# as_of cut handles through the date column below.
AS_OF_COLUMNS = {
    "marts": {
        "dim_users": "signup_date",
        "fct_kyc_attempts": "event_date",
        "fct_experiment_users": "assigned_date",
        "fct_activity_monthly": "month",
        "agg_kyc_daily": "decision_date",
        "agg_kyc_reject_reasons_monthly": "month",
        "agg_funnel_monthly": "signup_month",
        "agg_experiment_daily": "assigned_date",
        "agg_retention_cohorts": "activation_month",
        "stg_app_releases": "release_date",
    },
    "raw": {
        "users": "signup_ts",
        "funnel_events": "event_ts",
        "card_transactions": "txn_ts",
        "topups": "topup_ts",
        "incentive_payouts": "payout_ts",
        "subscriptions": "month",
        "experiment_assignments": "assigned_ts",
        "app_releases": "release_date",
    },
}

_READ_ONLY_START = re.compile(r"^\s*(select|with|describe|show|summarize|from|pivot|unpivot)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|create|drop|alter|attach|detach|copy|export|import|install|load|"
    r"pragma|set|call|checkpoint|vacuum|read_parquet|read_csv\w*|read_json\w*|glob)\b",
    re.I,
)


def build_raw_db() -> Path:
    """load data/raw/* into one duckdb file, once. ground_truth/ never goes in."""
    if RAW_DB.exists():
        return RAW_DB
    con = duckdb.connect(str(RAW_DB))
    for f in sorted(RAW_DIR.iterdir()):
        reader = "read_parquet" if f.suffix == ".parquet" else "read_csv_auto"
        con.sql(f"create table {f.stem} as select * from {reader}('{f.as_posix()}')")
    con.close()
    return RAW_DB


@dataclass
class QueryResult:
    sql: str
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0

    def to_tool_payload(self) -> dict:
        if not self.ok:
            return {"ok": False, "error": self.error}
        return {
            "ok": True,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "truncated_at_row_limit": self.truncated,
        }


def _jsonable(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "__float__") and not isinstance(v, (int, bool)):
        return round(float(v), 6)  # Decimal
    return v


class Warehouse:
    """one read-only session over either the marts or the raw events."""

    def __init__(self, source: str = "marts", as_of: str | None = None):
        assert source in ("marts", "raw")
        self.source, self.as_of = source, as_of
        path = MARTS_DB if source == "marts" else build_raw_db()
        self.con = duckdb.connect()
        self.con.sql(f"ATTACH '{path.as_posix()}' AS wh (READ_ONLY)")
        tables = [
            r[0]
            for r in self.con.sql(
                "select table_name from information_schema.tables "
                "where table_catalog = 'wh' and table_type = 'BASE TABLE' order by 1"
            ).fetchall()
        ]
        cuts = AS_OF_COLUMNS[source]
        for t in tables:
            where = ""
            if as_of and t in cuts:
                where = f" where {cuts[t]} <= timestamp '{as_of} 23:59:59'"
            self.con.sql(f"create view main.{t} as select * from wh.main.{t}{where}")
        self.tables = tables
        # one thread: unordered GROUP BY output then comes back in the same order
        # every time, which keeps the replay cache's request hashes stable
        self.con.sql("SET threads = 1")
        self.con.sql("SET enable_external_access = false")
        self.con.sql("SET lock_configuration = true")

    def run(self, sql: str, row_limit: int = ROW_LIMIT, timeout_s: float = TIMEOUT_S) -> QueryResult:
        sql = sql.strip().rstrip(";").strip()
        if ";" in sql:
            return QueryResult(sql, False, error="Only one statement per call is allowed.")
        if not _READ_ONLY_START.match(sql) or _FORBIDDEN.search(sql):
            return QueryResult(sql, False, error="Read-only access: only SELECT / WITH / DESCRIBE queries are allowed.")
        timer = threading.Timer(timeout_s, self.con.interrupt)
        t0 = time.perf_counter()
        timer.start()
        try:
            cur = self.con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(row_limit + 1)
        except duckdb.InterruptException:
            return QueryResult(sql, False, error=f"Query timed out after {timeout_s:.0f}s.",
                               elapsed_s=time.perf_counter() - t0)
        except duckdb.Error as e:
            # the exact DuckDB message goes back to the model so it can fix the query
            return QueryResult(sql, False, error=f"{type(e).__name__}: {e}", elapsed_s=time.perf_counter() - t0)
        finally:
            timer.cancel()
        return QueryResult(
            sql, True, cols,
            [[_jsonable(v) for v in r] for r in rows[:row_limit]],
            truncated=len(rows) > row_limit,
            elapsed_s=time.perf_counter() - t0,
        )

    # ---- schema context -------------------------------------------------

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.con.sql(f"describe {table}").fetchall()]

    def bare_schema(self) -> str:
        """table + column names and types, nothing else. what a text-to-SQL demo usually gets."""
        lines = []
        for t in self.tables:
            cols = ", ".join(f"{c} {ty}" for c, ty in self.columns(t))
            lines.append(f"{t}({cols})")
        return "\n".join(lines)

    def dbt_context(self, mart_name: str | None = None) -> str:
        """the documented semantic layer: model descriptions, column docs and
        metric definitions from schema.yml, plus the header comment of each
        export model's SQL (that's where its grain is explained)."""
        docs = _dbt_docs()
        names = [mart_name] if mart_name else self.tables
        unknown = [n for n in names if n not in self.tables]
        if unknown:
            return f"Unknown model {unknown[0]!r}. Available: {', '.join(self.tables)}"
        out = []
        for t in names:
            d = docs.get(t, {})
            out.append(f"## {t}")
            if d.get("description"):
                out.append(d["description"].strip())
            if d.get("sql_comment"):
                out.append(d["sql_comment"])
            col_docs = d.get("columns", {})
            for c, ty in self.columns(t):
                extra = col_docs.get(c, "")
                out.append(f"- {c} ({ty}){': ' + extra if extra else ''}")
            out.append("")
        if self.as_of:
            out.append(f"NOTE: every table is cut at {self.as_of}; treat that date as today.")
        return "\n".join(out)


_DOCS: dict | None = None


def _dbt_docs() -> dict:
    global _DOCS
    if _DOCS is not None:
        return _DOCS
    docs: dict[str, dict] = {}
    for yml in MODELS_DIR.rglob("schema.yml"):
        for m in (yaml.safe_load(yml.read_text()) or {}).get("models", []):
            cols = {}
            for c in m.get("columns", []):
                bits = []
                if c.get("description"):
                    bits.append(c["description"].strip())
                meta = c.get("meta", {}) or {}
                for name, spec in (meta.get("metrics") or {}).items():
                    s = f"metric `{name}` = {spec.get('type')}"
                    if spec.get("sql"):
                        s += f" of {spec['sql'].replace('${TABLE}.', '')}"
                    if spec.get("description"):
                        s += f" ({spec['description']})"
                    bits.append(s)
                if bits:
                    cols[c["name"]] = "; ".join(bits)
            docs[m["name"]] = {"description": m.get("description", ""), "columns": cols}
    for sql in MODELS_DIR.rglob("*.sql"):
        head = []
        for line in sql.read_text().splitlines():
            if not line.startswith("--"):
                break
            head.append(line[2:].strip())
        if head:
            docs.setdefault(sql.stem, {"columns": {}})["sql_comment"] = " ".join(head)
    _DOCS = docs
    return docs
