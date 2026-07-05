"""Bundle dashboard data for parkhouse.fr.

Combines the warehouse funnel aggregate with the analysis outputs into one
compact JSON the website imports. Keep it small - aggregates only, NEVER
row-level data, the site is public and nobody needs to see raw user rows.

Run:  python -m analysis.export_site [--site path/to/site/src/data/neobank.json]
"""

from __future__ import annotations

import argparse
import json
import shutil

from .common import OUT, connect, save_json

STAGES = ["signups", "email_verified", "kyc_started", "doc_submitted",
          "kyc_approved", "first_topup", "activated"]


def main(site_path: str | None) -> None:
    con = connect()

    funnel = con.sql(
        """
        select strftime(signup_month, '%Y-%m') as month, channel, device,
               n_signups, n_email_verified, n_kyc_started, n_doc_submitted,
               n_kyc_approved, n_first_topup, n_activated
        from agg_funnel_monthly order by 1, 2, 3
        """
    ).df()
    funnel_rows = [
        [r.month, r.channel, r.device,
         int(r.n_signups), int(r.n_email_verified), int(r.n_kyc_started),
         int(r.n_doc_submitted), int(r.n_kyc_approved), int(r.n_first_topup),
         int(r.n_activated)]
        for r in funnel.itertuples()
    ]

    kyc = json.loads((OUT / "kyc_incident.json").read_text())
    econ = json.loads((OUT / "retention_econ.json").read_text())
    exp = json.loads((OUT / "experiment.json").read_text())
    val = json.loads((OUT / "validation.json").read_text())

    payload = {
        "meta": {
            "window": ["2024-01", "2026-05"],
            "validation": {"passed": val["passed"], "total": val["total"]},
        },
        "funnel": {"stages": STAGES, "rows": funnel_rows},
        "kyc": {
            "series": kyc["weekly_series"],
            "incident": kyc["incident"],
            "attribution": kyc["attribution"],
            "impact": kyc["impact"],
            "reason_mix": kyc["reason_mix"],
        },
        "retention": {
            "by_channel": econ["retention_by_channel"],
            "churn_fit": econ["churn_fit"],
        },
        "economics": econ["economics"],
        "experiment": {
            "primary": exp["primary"],
            "adjustment": exp["adjustment"],
            "heterogeneity": exp["heterogeneity"],
            "margin_60d": exp["margin_60d"],
            "economics": exp["economics"],
            "verdict": exp["verdict"],
            "peeking": exp["peeking"],
            "srm": exp["srm"],
        },
    }
    path = save_json("site_data.json", payload)
    print(f"size: {path.stat().st_size / 1024:.0f} KB")  #DONT LET IT BLOAT WATCH IT
    if site_path:
        shutil.copy(path, site_path)
        print(f"copied to {site_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", default=None)
    main(p.parse_args().site)
