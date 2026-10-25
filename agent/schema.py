"""Structured output contract for every workflow's final answer."""

from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["SHIP", "DO_NOT_SHIP", "ROLLBACK_RELEASE", "PAUSE_ACQUISITION_CHANNEL", "NO_ACTION_NOISE"]


class SupportingMetric(BaseModel):
    name: str = Field(description="Exact metric name, e.g. 'android national_id KYC approval rate'.")
    baseline_value: float | None = Field(description="Value in the comparison/baseline period or arm, if any.")
    anomaly_value: float | None = Field(description="Value in the anomalous period, treatment arm or slice.")
    unit: str = Field(description="fraction, pp, EUR, EUR/user, count or months. Put dates in the text fields.")
    sql: str = Field(description="The SQL query or tool call that returned these values.")


class ProductTriageVerdict(BaseModel):
    incident_detected: bool
    root_cause_hypothesis: str
    supporting_metrics: list[SupportingMetric]
    recommended_action: Action
    reasoning_summary: str = Field(description="Numbers-first summary for a PM, at most 4 sentences.")


class SqlDraft(BaseModel):
    """baselines A and B answer in two calls: one query, then the verdict."""
    sql: str
    rationale: str


def strict_json_schema(model: type[BaseModel]) -> dict:
    """pydantic schema -> the subset structured outputs wants: every object
    closed (additionalProperties false), every property required, no titles."""
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema
