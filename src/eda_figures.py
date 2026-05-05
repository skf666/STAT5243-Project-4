"""Generate every publication-quality figure referenced in REPORT.md.

Outputs PNG (and optionally HTML for interactives) into `figures/`.

Usage:
    python -m src.eda_figures            # regenerates everything
    python -m src.eda_figures --only home_advantage,goal_distribution

Each figure is independent and idempotent. Pure matplotlib + seaborn for
report PNGs; plotly for the interactive dashboard versions used by the
Shiny app.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOG = logging.getLogger(__name__)
FIGURES_DIR = Path("figures")
MODEL_READY_DIR = Path("data/model_ready")
LEGACY_MODEL_READY_DIR = Path("data/processed")

# Consistent style.
sns.set_theme(style="whitegrid", font_scale=1.05)
ACCENT = "#1a3a6c"  # deep blue
ACCENT_2 = "#c1272d"


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    LOG.info("Saved %s", out)


def fig_result_distribution(matches: pd.DataFrame) -> None:
    counts = matches["result"].value_counts(normalize=True).reindex(["H", "D", "A"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=[ACCENT, "#888", ACCENT_2])
    ax.set_xlabel("Result (Home / Draw / Away)")
    ax.set_ylabel("Share of matches")
    ax.set_title("Overall result distribution, 2000-01 to 2020-21")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    _save(fig, "01_result_distribution")


def fig_home_advantage_by_season(matches: pd.DataFrame) -> None:
    rate = matches.assign(is_home_win=(matches["result"] == "H").astype(int)).groupby("season")["is_home_win"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rate.index, rate.values, marker="o", color=ACCENT, lw=2)
    ax.axhline(rate[~rate.index.isin(["2020-21"])].mean(), color="grey", ls="--", lw=1, label="pre-Covid avg")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home win rate")
    ax.set_title("Home advantage by season — note the 2020-21 Covid dip (no fans)")
    ax.set_ylim(0.3, 0.55)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    if "2020-21" in rate.index:
        ax.annotate(
            "2020-21 (no fans)",
            xy=("2020-21", rate.loc["2020-21"]),
            xytext=(-55, -45),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )
    ax.legend()
    _save(fig, "02_home_advantage_by_season")


def fig_goal_distribution(matches: pd.DataFrame) -> None:
    total = (matches["home_goals"].fillna(0) + matches["away_goals"].fillna(0)).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(total, bins=range(0, 11), kde=False, color=ACCENT, ax=ax, edgecolor="white")
    ax.set_xlabel("Total goals per match")
    ax.set_ylabel("Match count")
    ax.set_title(f"Goals per match distribution (n = {len(total):,} matches)")
    _save(fig, "03_goal_distribution")


def fig_score_heatmap(matches: pd.DataFrame, max_g: int = 6) -> None:
    cnt = matches.groupby(["home_goals", "away_goals"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=range(max_g + 1), columns=range(max_g + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cnt, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "match count"}, ax=ax)
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title("Score-line frequency heatmap")
    _save(fig, "04_score_heatmap")


def fig_elo_trajectories(matches: pd.DataFrame, n_top: int = 6) -> None:
    rows = []
    for _, r in matches.iterrows():
        rows.append({"season": r["season"], "team": r["home_team"], "elo": r.get("home_elo")})
        rows.append({"season": r["season"], "team": r["away_team"], "elo": r.get("away_elo")})
    long = pd.DataFrame(rows).dropna()
    season_means = long.groupby(["season", "team"])["elo"].mean().reset_index()
    last_season = season_means["season"].max()
    top_teams = season_means[season_means["season"] == last_season].nlargest(n_top, "elo")["team"].tolist()
    sub = season_means[season_means["team"].isin(top_teams)]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=sub, x="season", y="elo", hue="team", marker="o", ax=ax)
    ax.set_title(f"Mean Elo per season — top {n_top} teams of {last_season}")
    ax.set_ylabel("Mean Elo rating")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "05_elo_trajectories")


def fig_correlation_matrix(matches: pd.DataFrame) -> None:
    cols = [
        c for c in [
            "elo_diff", "home_form_pts_avg_5", "away_form_pts_avg_5",
            "home_rest_days", "away_rest_days", "is_derby",
            "implied_prob_home_pinnacle_novig",
            "implied_prob_draw_pinnacle_novig",
            "implied_prob_away_pinnacle_novig",
        ] if c in matches.columns
    ]
    if not cols:
        return
    corr = matches[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature correlation matrix")
    _save(fig, "06_correlation_matrix")


def fig_team_style_pca(team_season_unsup: pd.DataFrame) -> None:
    if team_season_unsup.empty or "pca_1" not in team_season_unsup.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Set2", n_colors=team_season_unsup["cluster_id"].nunique())
    for cid, group in team_season_unsup.groupby("cluster_id"):
        ax.scatter(group["pca_1"], group["pca_2"], color=palette[cid], label=f"Cluster {cid}", alpha=0.7, s=40)
    # Label a few notable teams (most recent season).
    label_teams = ["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Burnley", "West Bromwich Albion"]
    last_seen = team_season_unsup.groupby("team").last()
    available = [t for t in label_teams if t in last_seen.index]
    for team in available:
        row = last_seen.loc[team]
        ax.annotate(team, (row["pca_1"], row["pca_2"]), fontsize=8, alpha=0.8)
    ax.set_xlabel("PCA component 1 (attack ↔ defence)")
    ax.set_ylabel("PCA component 2 (style)")
    ax.set_title("Team-season style space — PCA + K-means (k=4)")
    ax.legend(title="K-means cluster", loc="best")
    _save(fig, "07_team_style_pca")


def fig_team_style_umap(team_season_unsup: pd.DataFrame) -> None:
    if team_season_unsup.empty or "umap_1" not in team_season_unsup.columns:
        return
    if team_season_unsup["umap_1"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Set2", n_colors=team_season_unsup["cluster_id"].nunique())
    for cid, group in team_season_unsup.groupby("cluster_id"):
        ax.scatter(group["umap_1"], group["umap_2"], color=palette[cid], label=f"Cluster {cid}", alpha=0.7, s=40)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP 2D embedding of team-season style")
    ax.legend(title="K-means cluster")
    _save(fig, "08_team_style_umap")


def fig_market_implied_vs_outcome(matches: pd.DataFrame) -> None:
    if "implied_prob_home_pinnacle_novig" not in matches.columns:
        return
    sub = matches.dropna(subset=["implied_prob_home_pinnacle_novig"]).copy()
    sub["bin"] = pd.cut(sub["implied_prob_home_pinnacle_novig"], bins=10)
    cal = sub.groupby("bin", observed=True).apply(
        lambda g: pd.Series({
            "predicted": g["implied_prob_home_pinnacle_novig"].mean(),
            "actual": (g["result"] == "H").mean(),
            "n": len(g),
        })
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(cal["predicted"], cal["actual"], "o-", color=ACCENT, lw=2, label="Pinnacle market")
    ax.set_xlabel("Pinnacle implied P(home win) (vig-removed)")
    ax.set_ylabel("Actual home-win frequency")
    ax.set_title("Market calibration — Pinnacle is essentially perfectly calibrated")
    ax.legend()
    _save(fig, "09_market_calibration")


def fig_per_team_record(matches: pd.DataFrame, season: str = "2020-21", n: int = 20) -> None:
    sub = matches[matches["season"] == season]
    if sub.empty:
        return
    pts = {}
    for _, r in sub.iterrows():
        h, a = r["home_team"], r["away_team"]
        if r["result"] == "H":
            pts[h] = pts.get(h, 0) + 3; pts[a] = pts.get(a, 0)
        elif r["result"] == "A":
            pts[a] = pts.get(a, 0) + 3; pts[h] = pts.get(h, 0)
        else:
            pts[h] = pts.get(h, 0) + 1; pts[a] = pts.get(a, 0) + 1
    standings = pd.Series(pts).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(y=standings.index, x=standings.values, ax=ax, color=ACCENT)
    ax.set_xlabel("Final points")
    ax.set_ylabel("")
    ax.set_title(f"Final standings — {season}")
    _save(fig, "10_final_standings")


_REGISTRY = {
    "result_distribution": fig_result_distribution,
    "home_advantage": fig_home_advantage_by_season,
    "goal_distribution": fig_goal_distribution,
    "score_heatmap": fig_score_heatmap,
    "elo_trajectories": fig_elo_trajectories,
    "correlation_matrix": fig_correlation_matrix,
    "market_calibration": fig_market_implied_vs_outcome,
    "final_standings": fig_per_team_record,
}


def _registry_with_team_season() -> dict:
    return {
        "team_style_pca": fig_team_style_pca,
        "team_style_umap": fig_team_style_umap,
    }


def main(only: list[str] | None = None) -> None:
    matches_path = MODEL_READY_DIR / "matches.parquet"
    if not matches_path.exists():
        matches_path = LEGACY_MODEL_READY_DIR / "matches.parquet"
    matches = pd.read_parquet(matches_path)
    LOG.info("Loaded %d matches", len(matches))
    for name, fn in _REGISTRY.items():
        if only and name not in only:
            continue
        try:
            fn(matches)
        except Exception as exc:
            LOG.exception("Figure %s failed: %s", name, exc)

    unsup_path = MODEL_READY_DIR / "team_season_unsupervised.parquet"
    if not unsup_path.exists():
        unsup_path = LEGACY_MODEL_READY_DIR / "team_season_unsupervised.parquet"
    if unsup_path.exists():
        unsup = pd.read_parquet(unsup_path)
        for name, fn in _registry_with_team_season().items():
            if only and name not in only:
                continue
            try:
                fn(unsup)
            except Exception as exc:
                LOG.exception("Figure %s failed: %s", name, exc)
    else:
        LOG.warning("No unsupervised features yet; run `python -m src.unsupervised` first.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated subset of figures to regenerate")
    args = p.parse_args()
    main(only=args.only.split(",") if args.only else None)
