"""The agent's stats tool must agree with the analysis layer, and the grader
must accept the pipeline's own numbers and reject made-up ones.

    python -m evals.test_stats      (needs analysis/out from scripts/run_pipeline.py)
"""

from __future__ import annotations

import json

from agent.stats import ab_test, changepoint, channel_unit_economics
from agent.warehouse import ROOT, Warehouse

from .cases import CASES
from .grading import fabricated_metrics, grounded, hit, _observed

OUT = ROOT / "analysis" / "out"


def main() -> None:
    wh = Warehouse("marts")
    kyc = json.loads((OUT / "kyc_incident.json").read_text())["incident"]
    econ = json.loads((OUT / "retention_econ.json").read_text())
    exp = json.loads((OUT / "experiment.json").read_text())

    cp = changepoint(wh, "android", "national_id")
    for k in ("break_date", "recovery_date", "level_before", "level_during", "level_after"):
        assert cp[k] == kyc[k], (k, cp[k], kyc[k])
    ue = channel_unit_economics(wh)["by_channel"]
    for ch, e in econ["economics"].items():
        assert ue[ch]["ltv_36m_eur"] == e["ltv_36m_eur"] and ue[ch]["cac_per_activated_eur"] == e["cac_eur"], ch
        assert ue[ch]["implied_monthly_churn"] == econ["churn_fit"][ch]["implied_monthly_churn"], ch
    ab = ab_test(wh, "channel")
    assert ab["overall"]["delta_pp"] == exp["primary"]["delta_pp"]
    assert ab["overall"]["margin_60d_delta_net_of_bonus_eur"] == exp["margin_60d"]["delta_eur"]
    for ch, h in exp["heterogeneity"]["by_channel"].items():
        assert ab["by_channel"][ch]["delta_pp"] == h["delta_pp"], ch
    print("stats tool matches analysis/out: ok")

    # noise slices really are flat
    assert not changepoint(wh, "ios", "driving_licence")["break_detected"]

    # grader: a verdict quoting the pipeline's numbers hits every target
    for case in CASES:
        for t in case["targets"]:
            text = t["value"] if t["kind"] != "number" else f"{t['value']}"
            v = {"root_cause_hypothesis": str(text), "reasoning_summary": "",
                 "supporting_metrics": [], "recommended_action": "NO_ACTION_NOISE",
                 "incident_detected": False}
            assert hit(t, v), (case["id"], t["label"])
    obs = _observed(['{"rows": [[0.888, 0.686], [5065, 5089]]}'])
    assert grounded(0.686, obs) and grounded(68.6, obs) and grounded(-20.2, obs)  # pp delta
    assert not grounded(0.731, obs)
    rec = {"verdict": {"supporting_metrics": [{"name": "x", "baseline_value": 0.5,
                                                "anomaly_value": 0.4, "unit": "", "sql": ""}]},
           "tool_outputs": ['{"rows": [[0.5]]}']}
    assert [f["value"] for f in fabricated_metrics(rec)] == [0.4]
    print("grader sanity checks: ok")


if __name__ == "__main__":
    main()
