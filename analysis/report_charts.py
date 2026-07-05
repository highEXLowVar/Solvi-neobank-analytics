"""Draws the two README figures for the follow-up analyses.

Everything comes from analysis/out, nothing is recomputed here. The oracle
curve is exported by validate_truth, so this file never reads ground truth.

Run:  python -m analysis.report_charts   (after validate_truth)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import OUT, ROOT

ASSETS = ROOT / "assets"
ACCENT = "#3457c0"
GREY = "#8a8a8a"
INK = "#2b2b2b"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def cuped_figure() -> None:
    c = json.loads((OUT / "cuped.json").read_text())
    prior = c["comparison"]
    rows = [
        ("unadjusted", prior["unadjusted"]["delta_pp"], prior["unadjusted"]["se_pp"]),
        ("regression-adjusted", prior["regression_adjustment"]["delta_pp"], prior["regression_adjustment"]["se_pp"]),
        ("CUPED / CUPAC", prior["cuped_cupac"]["delta_pp"], prior["cuped_cupac"]["se_pp"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (label, d, se) in enumerate(rows):
        y = len(rows) - 1 - i
        colour = ACCENT if "CUPED" in label else GREY
        ax.errorbar(d, y, xerr=1.96 * se, fmt="o", color=colour,
                    capsize=3, markersize=5, elinewidth=1.4)
        ax.annotate(f"{d:.2f}pp ± {1.96 * se:.2f}", (d, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=colour)
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel("effect on 14-day conversion (pp, 95% CI)")
    vr = c["primary_conv_14d"]["variance_reduction"]
    rho2 = round(c["primary_conv_14d"]["rho_y_x"] ** 2, 3)
    ax.set_title(f"{vr:.1%} variance reduction, matching rho2 = {rho2}",
                 fontsize=10, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "cuped_ci.png", dpi=160)
    plt.close(fig)


def uplift_figure() -> None:
    u = json.loads((OUT / "uplift.json").read_text())
    oracle = json.loads((OUT / "oracle_curve.json").read_text())
    groups = u["policies"]["A_rank_by_uplift"]["groups"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.1))

    xs = [g["group"] for g in groups]
    ys = [g["uplift_obs_pp"] for g in groups]
    es = [1.96 * g["uplift_se_pp"] for g in groups]
    ax1.axhline(0, color=GREY, lw=0.7)
    ax1.axhline(u["avg_uplift_pp"], color=GREY, lw=1, ls="--")
    ax1.annotate(f"average lift {u['avg_uplift_pp']:.1f}pp", (10.3, u["avg_uplift_pp"]),
                 fontsize=8.5, color=GREY, ha="right", va="bottom")
    ax1.errorbar(xs, ys, yerr=es, fmt="o", color=ACCENT, capsize=3,
                 markersize=4.5, elinewidth=1.2)
    ax1.set_xticks(xs)
    ax1.set_xlabel("decile by predicted lift (1 = model's favourite)")
    ax1.set_ylabel("observed lift (pp, 95% CI)")
    ax1.set_title("Flat: no targetable signal in the ranking", fontsize=10, loc="left")

    fr = oracle["frac"]
    cp = [v / 1000 for v in oracle["cum_profit_eur"]]
    ax2.axhline(0, color=GREY, lw=0.7)
    ax2.plot(fr, cp, color=ACCENT, lw=1.8)
    ax2.set_xlabel("fraction of users targeted, ranked by TRUE effect")
    ax2.set_ylabel("cumulative profit (kEUR)")
    ax2.set_title("Even an oracle never breaks even", fontsize=10, loc="left")
    ax2.annotate(f"best possible slice: EUR{oracle['best_eur']:.0f} (a single user)",
                 (0.03, -1), fontsize=8.5, color=INK, va="top")

    fig.tight_layout()
    fig.savefig(ASSETS / "uplift_targeting.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    cuped_figure()
    uplift_figure()
    print(f"wrote {ASSETS.relative_to(ROOT)}/cuped_ci.png and uplift_targeting.png")


if __name__ == "__main__":
    main()
