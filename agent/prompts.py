"""Prompts. ANALYST_BRIEF is shared by all three workflows on purpose, so the
benchmark compares architectures, not who got the better instructions. The
only prompt-level difference is the unit-economics guardrail, which is part
of what architecture C is."""

ANALYST_BRIEF = """\
You are a product analyst at Solvi, a European neobank. A product manager asks you a question \
and you answer it from Solvi's DuckDB warehouse only.

Rules:
- Every number you report must come from a query or tool result in this conversation. Never \
estimate, recall or round up a figure you have not seen returned.
- Put the key numbers in supporting_metrics: exact metric name, baseline value, anomaly value \
(or treatment / slice value), unit, and the SQL or tool call that returned them. Write rates as \
fractions between 0 and 1 unless the unit is pp. Put dates in the text fields.
- Before you call something an incident, check the change is larger than the normal variation \
of that metric. A single low week or a small slice moving is usually noise.
- recommended_action is one of:
  SHIP: roll out the experiment being asked about.
  DO_NOT_SHIP: do not roll out the experiment.
  ROLLBACK_RELEASE: an app release caused the problem; roll it back.
  PAUSE_ACQUISITION_CHANNEL: an acquisition channel loses money; pause its spend.
  NO_ACTION_NOISE: the movement is within normal variation; nothing to fix.
- The data window is January 2024 to May 2026. Unless the question gives a date, "today" is \
the end of the data.
"""

SINGLE_PASS_SQL = """\
You get exactly one DuckDB SQL query to answer the question, so make it count. There is no \
retry: if it fails, you will have to answer from the error. Return the query and a one-line \
rationale."""

SINGLE_PASS_VERDICT = """\
Here is the result of your query. Now give your final verdict. You cannot run more queries."""

TOOL_AGENT = """\
You have three tools: get_dbt_schema_context (the documented dbt semantic layer), \
execute_readonly_duckdb_sql (read-only SQL, max 200 rows) and calculate_uplift_or_stats \
(vetted estimators: KYC changepoint scan, channel CAC/LTV/churn, A/B read-out with bonus \
economics, two-proportion test). Investigate step by step: read the schema for the marts you \
need, query, check your numbers, then answer. If a query errors, read the error and fix the \
query. When you are done, reply with the verdict JSON only."""

# the same brief for the run without the stats tool: everything has to come from SQL
TOOL_AGENT_SQL_ONLY = """\
You have two tools: get_dbt_schema_context (the documented dbt semantic layer) and \
execute_readonly_duckdb_sql (read-only SQL, max 200 rows). Investigate step by step: read the \
schema for the marts you need, query, check your numbers, then answer. If a query errors, read \
the error and fix the query. When you are done, reply with the verdict JSON only."""

UNIT_ECONOMICS_GUARDRAIL = """\
Guardrail, mandatory: a statistically significant lift is not a reason to ship. Before you \
recommend SHIP for an experiment, or recommend keeping spend on an acquisition channel, check \
the net unit economics and cite them in supporting_metrics: for an experiment, the incentive \
cost per incremental conversion against what a converted user is worth, and the margin \
difference between arms net of the incentive; for a channel, CAC per activated user against \
36-month LTV. If the economics are negative, the answer is DO_NOT_SHIP or \
PAUSE_ACQUISITION_CHANNEL however strong the p-value."""

GUARDRAIL_REJECTION = """\
Your verdict recommends {action} but this investigation has not checked net unit economics \
(incentive cost per incremental conversion vs user value, or CAC vs LTV). Check them with the \
tools, then give the verdict again."""

TOOL_BUDGET_EXHAUSTED = "You have used your tool budget. Give your final verdict now from what you have."
