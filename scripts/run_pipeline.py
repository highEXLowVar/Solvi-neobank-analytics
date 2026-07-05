"""Run the whole pipeline end to end: simulate -> warehouse -> analyse -> validate.

Usage:  python scripts/run_pipeline.py
Takes ~1 minute. Requires the venv with requirements.txt installed, obviously.
If a step fails halfway through, just fix it and rerun the whole thing, this
isnt set up to resume from the middle and its not worth building that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# dbt lives in the same venv as whatever python is running this - if this path
# guess is wrong on your machine just fall back to whatever's on PATH
_dbt = Path(sys.executable).parent / ("dbt.exe" if sys.platform == "win32" else "dbt")
DBT = str(_dbt) if _dbt.exists() else "dbt"

STEPS: list[tuple[str, list[str], Path]] = [
    ("simulate", [sys.executable, "-m", "simulator.run"], ROOT),
    ("dbt build", [DBT, "build", "--profiles-dir", "."], ROOT / "warehouse"),
    ("kyc incident", [sys.executable, "-m", "analysis.kyc_incident"], ROOT),
    ("retention & economics", [sys.executable, "-m", "analysis.retention_economics"], ROOT),
    ("experiment", [sys.executable, "-m", "analysis.experiment_eval"], ROOT),
    ("cuped / cupac", [sys.executable, "-m", "analysis.cuped"], ROOT),
    ("uplift targeting", [sys.executable, "-m", "analysis.uplift"], ROOT),
    ("validate vs ground truth", [sys.executable, "-m", "analysis.validate_truth"], ROOT),
    ("export dashboard data", [sys.executable, "-m", "analysis.export_site"], ROOT),
    ("readme figures", [sys.executable, "-m", "analysis.report_charts"], ROOT),
]


def main() -> None:
    for name, cmd, cwd in STEPS:
        print(f"\n=== {name} " + "=" * max(0, 56 - len(name)))
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            raise SystemExit(f"step failed: {name}")
    print("\npipeline complete, all checks green")


if __name__ == "__main__":
    main()
