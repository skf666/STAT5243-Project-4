"""Generate figures/10_mc_title_probabilities.png — the Monte Carlo
title / top-4 / relegation probability chart used on oral slide 10.

Run from the repo root:
    python3 slides/_make_mc_figure.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Repo root + sys.path so `from src...` works whether run from repo root or slides/.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.season_sim import simulate_season  # noqa: E402
from src.train import available_features  # noqa: E402

# Columbia palette
NAVY = "#0B2545"
SKY = "#B9D9EB"
SKY_DK = "#7BA9C8"
CORAL = "#E07A5F"
GOLD = "#F2C14E"
INK = "#1a1a2e"
WHITE = "#FFFFFF"

PROCESSED = REPO / "data" / "processed" / "matches.parquet"
MODEL = REPO / "results" / "model_random_forest.joblib"
OUT = REPO / "figures" / "10_mc_title_probabilities.png"


def main() -> None:
    print(f"Loading {PROCESSED}")
    df = pd.read_parquet(PROCESSED)
    test = df[df["season"] == "2020-21"].copy().reset_index(drop=True)
    print(f"  test season has {len(test)} matches, {len(set(test['home_team']) | set(test['away_team']))} teams")

    feats = available_features(df)
    print(f"  using {len(feats)} features")

    print(f"Loading {MODEL}")
    model = joblib.load(MODEL)
    proba = model.predict_proba(test[feats])
    test["p_home"] = proba[:, 0]
    test["p_draw"] = proba[:, 1]
    test["p_away"] = proba[:, 2]

    print("Running 2000-iter Monte Carlo (this takes ~30s)...")
    sim = simulate_season(test[["home_team", "away_team", "p_home", "p_draw", "p_away"]],
                          n_iter=2000, seed=20260427)
    sim = sim.sort_values("expected_rank").reset_index(drop=True)
    print(f"  simulated {len(sim)} teams; top expected rank = {sim['expected_rank'].min():.2f}")

    # Build the chart: horizontal bar of expected_rank with rank-percentile error bars,
    # color-coded by championship/top4/relegation probability tier.
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150, facecolor=WHITE)
    ax.set_facecolor(WHITE)

    n = len(sim)
    y = np.arange(n)[::-1]  # rank 1 at top
    means = sim["expected_rank"].values
    p05 = sim["rank_p05"].values
    p95 = sim["rank_p95"].values

    # Error bars (left = better rank, right = worse rank).
    err_left = means - p05
    err_right = p95 - means

    # Color: gold for any team with non-trivial title prob (>=2%), sky for top-4 (>=20%),
    # coral for relegation candidates (>=20%), navy otherwise.
    title_p = sim["title_prob"].values
    top4_p = sim["top4_prob"].values
    releg_p = sim["relegation_prob"].values
    colors = []
    for i in range(n):
        if title_p[i] >= 0.02:
            colors.append(GOLD)
        elif top4_p[i] >= 0.20:
            colors.append(SKY_DK)
        elif releg_p[i] >= 0.20:
            colors.append(CORAL)
        else:
            colors.append(NAVY)

    ax.errorbar(means, y, xerr=[err_left, err_right],
                fmt="none", ecolor=SKY_DK, alpha=0.6, capsize=3, capthick=1.0, lw=1.0)
    ax.scatter(means, y, c=colors, s=110, edgecolor=NAVY, linewidth=0.7, zorder=3)

    # Annotate each bar with team name + key prob.
    for i in range(n):
        team = sim["team"].iloc[i]
        label_extra = ""
        if title_p[i] >= 0.02:
            label_extra = f"  ·  P(title)={title_p[i]:.0%}"
        elif top4_p[i] >= 0.20:
            label_extra = f"  ·  P(top-4)={top4_p[i]:.0%}"
        elif releg_p[i] >= 0.20:
            label_extra = f"  ·  P(rel.)={releg_p[i]:.0%}"
        ax.text(means[i] + max(err_right[i], 0.5) + 0.3, y[i],
                f"{team}{label_extra}",
                va="center", ha="left", fontsize=9, color=INK)

    ax.set_xlabel("Predicted final rank   (mean ± 5/95 percentile, 2,000 Monte Carlo seasons)",
                  fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(rank)}" for rank in np.arange(1, n + 1)], color=INK, fontsize=8)
    ax.set_ylabel("Predicted rank order", fontsize=10, color=INK)
    ax.set_xlim(0, n + 7)  # leave room on the right for labels
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_xaxis()  # reverse so rank 1 visually leftmost
    ax.invert_xaxis()
    ax.grid(True, axis="x", color=SKY, alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(SKY_DK)
    ax.spines["bottom"].set_color(SKY_DK)

    # Title + legend.
    fig.suptitle("Monte Carlo 2020-21 final standings — Random Forest probabilities, 2,000 simulations",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.97, x=0.04, ha="left")

    # Inline legend.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=GOLD, edgecolor=NAVY, label="P(title) ≥ 2%"),
        Patch(facecolor=SKY_DK, edgecolor=NAVY, label="P(top-4) ≥ 20%"),
        Patch(facecolor=CORAL, edgecolor=NAVY, label="P(relegation) ≥ 20%"),
        Patch(facecolor=NAVY, edgecolor=NAVY, label="mid-table"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8, ncol=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=WHITE)
    print(f"Saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
