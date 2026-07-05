"""grab-bag of helpers the analysis scripts all need, didnt want to repeat this 6 times"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "solvi.duckdb"
OUT = ROOT / "analysis" / "out"


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)

#LOOK AT THIS IF DOESNT WORK
def _np_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def save_json(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, default=_np_safe))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """wilson score interval, dont use the naive normal approx it falls apart near 0/1"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    halfWidth2 = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - halfWidth2, centre + halfWidth2)
