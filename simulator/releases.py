"""App release calendar and per-submission app version assignment."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as C


def release_calendar() -> pd.DataFrame:
    """deterministic release schedule per platform.

    the android release closest to INCIDENT_RELEASE_DATE gets snapped to it
    exactly (that's the version with the KYC image-compression bug), and the
    release two slots later gets snapped to HOTFIX_RELEASE_DATE. bit dodge but
    it guarantees the incident actually lands on a real release.
    """
    rows = []
    for platform, first in (("android", C.ANDROID_FIRST_RELEASE), ("ios", C.IOS_FIRST_RELEASE)):
        dates = [first]
        for gap in itertools.cycle(C.RELEASE_GAPS):
            nxt = dates[-1] + np.timedelta64(gap, "D")
            if nxt > C.END_DATE:
                break
            dates.append(nxt)
        dates = np.array(dates, dtype="datetime64[D]")
        if platform == "android":
            inc_idx = int(np.argmin(np.abs((dates - C.INCIDENT_RELEASE_DATE).astype(int))))
            dates[inc_idx] = C.INCIDENT_RELEASE_DATE
            dates[inc_idx + 1] = C.INCIDENT_RELEASE_DATE + np.timedelta64(26, "D")
            dates[inc_idx + 2] = C.HOTFIX_RELEASE_DATE
        major = 5 if platform == "android" else 8
        for i, d in enumerate(dates):
            rows.append({"platform": platform, "version": f"{major}.{i}", "release_date": d})
    cal = pd.DataFrame(rows)
    cal["release_date"] = cal["release_date"].astype("datetime64[ns]")
    return cal


def incident_versions(cal: pd.DataFrame) -> list[str]:
    """Android versions carrying the bug: incident release up to (excl.) hotfix."""
    a = cal[cal["platform"] == "android"].sort_values("release_date")
    mask = (a["release_date"] >= pd.Timestamp(C.INCIDENT_RELEASE_DATE)) & (
        a["release_date"] < pd.Timestamp(C.HOTFIX_RELEASE_DATE)
    )
    return a.loc[mask, "version"].tolist()


def version_at(
    rng: np.random.Generator, cal: pd.DataFrame, platform: np.ndarray, ts: np.ndarray
) -> np.ndarray:
    """app version in use at each timestamp - mostly latest, sometimes 1-2 releases behind"""
    out = np.empty(len(ts), dtype=object)
    behind = rng.choice(len(C.VERSION_ADOPTION), len(ts), p=C.VERSION_ADOPTION)  # laggards, basically
    for plat in ("android", "ios"):
        sub = cal[cal["platform"] == plat].sort_values("release_date")
        rel_dates = sub["release_date"].to_numpy()
        versions = sub["version"].to_numpy()
        mask = platform == plat
        idx = np.searchsorted(rel_dates, ts[mask], side="right") - 1
        idx = np.maximum(idx - behind[mask], 0)
        out[mask] = versions[idx]
    return out
