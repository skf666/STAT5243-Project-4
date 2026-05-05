"""Unsupervised learning over team-season aggregate stats.

Two roles:
1. **EDA / interpretability** — visualize team playing styles in a 2D space
   (PCA / UMAP / t-SNE), label by season or manager, look for clusters.
2. **Feature engineering** — output per (season, team) cluster IDs + low-dim
   PCA components that we attach back to matches as predictors. The cluster
   ID captures "what kind of team is this" beyond raw match-level form.

Inputs: a per-team-per-season aggregate (built in `build_team_season_table`).
Outputs:
- `data/model_ready/team_season_unsupervised.parquet` (one row per (season, team)
  with `pca_1..pca_5`, `cluster_id`, `umap_1`, `umap_2`)
- Figures saved to `figures/` for the report.

Leakage-safety: the unsupervised model is fit on the **training span only**
(seasons strictly before the test season 2020-21), then transform applied to
all rows. We DO use prior-season aggregates to label the current season
(matches a 2020-21 match's home team to its 2019-20 cluster).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
MODEL_READY_DIR = Path("data/model_ready")
LEGACY_MODEL_READY_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")


# Columns we aggregate per team-season to build the style vector.
_AGG_COLS = [
    "home_goals", "away_goals",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellows", "away_yellows",
    "home_reds", "away_reds",
]


def _team_season_agg(matches: pd.DataFrame) -> pd.DataFrame:
    """Pivot match-level data to (season, team) -> per-90 aggregate stats."""
    pieces = []
    for _, row in matches.iterrows():
        for venue, team_col, opp_col in [("H", "home_team", "away_team"), ("A", "away_team", "home_team")]:
            team = row[team_col]
            if not isinstance(team, str):
                continue
            d = {"season": row["season"], "team": team, "venue": venue}
            d["goals_for"] = row.get("home_goals" if venue == "H" else "away_goals")
            d["goals_against"] = row.get("away_goals" if venue == "H" else "home_goals")
            d["shots_for"] = row.get("home_shots" if venue == "H" else "away_shots")
            d["shots_against"] = row.get("away_shots" if venue == "H" else "home_shots")
            d["sot_for"] = row.get("home_shots_target" if venue == "H" else "away_shots_target")
            d["corners_for"] = row.get("home_corners" if venue == "H" else "away_corners")
            d["fouls_for"] = row.get("home_fouls" if venue == "H" else "away_fouls")
            d["yellows_for"] = row.get("home_yellows" if venue == "H" else "away_yellows")
            d["reds_for"] = row.get("home_reds" if venue == "H" else "away_reds")
            pieces.append(d)
    long = pd.DataFrame(pieces)
    agg = long.groupby(["season", "team"]).mean(numeric_only=True).reset_index()
    return agg


def fit_unsupervised(
    team_season: pd.DataFrame,
    *,
    train_seasons: list[str],
    n_pca: int = 5,
    n_clusters: int = 4,
    random_state: int = 20260426,
) -> tuple[Pipeline, KMeans, pd.DataFrame]:
    """Fit PCA + K-means on the training-season slice, transform all rows."""
    train = team_season[team_season["season"].isin(train_seasons)].copy()
    feat_cols = [c for c in train.columns if c not in {"season", "team"}]
    LOG.info("Fitting unsupervised models on %d rows × %d features", len(train), len(feat_cols))

    pre = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    X_train = pre.fit_transform(train[feat_cols])
    pca = PCA(n_components=min(n_pca, len(feat_cols)), random_state=random_state)
    pca.fit(X_train)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(pca.transform(X_train))

    # Transform every (season, team) row.
    full = team_season.copy()
    X_full = pre.transform(full[feat_cols])
    pca_full = pca.transform(X_full)
    for i in range(pca_full.shape[1]):
        full[f"pca_{i + 1}"] = pca_full[:, i]
    full["cluster_id"] = km.predict(pca_full)

    LOG.info("PCA explained variance: %s", [round(x, 3) for x in pca.explained_variance_ratio_])
    return Pipeline([("pre", pre), ("pca", pca)]), km, full


def add_umap(team_season: pd.DataFrame, *, random_state: int = 20260426) -> pd.DataFrame:
    """Optional 2D UMAP embedding for visualization (only — not for modeling)."""
    try:
        import umap

        feat_cols = [c for c in team_season.columns if c.startswith("pca_")]
        if not feat_cols:
            return team_season
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=random_state)
        coords = reducer.fit_transform(team_season[feat_cols].fillna(0))
        out = team_season.copy()
        out["umap_1"] = coords[:, 0]
        out["umap_2"] = coords[:, 1]
        return out
    except Exception as exc:
        LOG.warning("UMAP failed: %s", exc)
        team_season["umap_1"] = np.nan
        team_season["umap_2"] = np.nan
        return team_season


def attach_to_matches(matches: pd.DataFrame, team_season_unsup: pd.DataFrame) -> pd.DataFrame:
    """Join per-(season, team) PCA + cluster ID into matches as home_/away_ columns.

    Leakage-safe: we join PRIOR-SEASON unsupervised features (so a 2020-21
    match uses each team's 2019-20 style vector to describe its prior style).
    """
    out = matches.copy()
    out["prev_season"] = out["season"].map(
        lambda s: f"{int(s.split('-')[0]) - 1}-{int(s.split('-')[0]) % 100:02d}"
    )
    keep = [c for c in team_season_unsup.columns if c.startswith("pca_") or c == "cluster_id"]
    rf = team_season_unsup.rename(columns={"season": "prev_season"})[["prev_season", "team"] + keep]

    home = rf.add_prefix("home_").rename(
        columns={"home_team": "home_team", "home_prev_season": "prev_season"}
    )
    away = rf.add_prefix("away_").rename(
        columns={"away_team": "away_team", "away_prev_season": "prev_season"}
    )

    out = out.merge(home, on=["home_team", "prev_season"], how="left")
    out = out.merge(away, on=["away_team", "prev_season"], how="left")
    return out.drop(columns=["prev_season"])


def build_unsupervised(matches: pd.DataFrame) -> pd.DataFrame:
    """Run the full unsupervised pipeline. Returns the per-(season, team) frame."""
    LOG.info("Aggregating to team-season vectors from %d matches", len(matches))
    team_season = _team_season_agg(matches)
    LOG.info("Got %d team-season rows", len(team_season))
    train_seasons = [s for s in sorted(matches["season"].unique()) if s not in {"2019-20", "2020-21"}]
    _, _, team_season_full = fit_unsupervised(team_season, train_seasons=train_seasons)
    team_season_full = add_umap(team_season_full)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    team_season_full.to_parquet(MODEL_READY_DIR / "team_season_unsupervised.parquet", index=False)
    LOG.info("Wrote %d rows to %s", len(team_season_full), MODEL_READY_DIR / "team_season_unsupervised.parquet")
    return team_season_full


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    matches_path = MODEL_READY_DIR / "matches.parquet"
    if not matches_path.exists():
        matches_path = LEGACY_MODEL_READY_DIR / "matches.parquet"
    matches = pd.read_parquet(matches_path)
    out = build_unsupervised(matches)
    print(out.groupby("cluster_id")[["pca_1", "pca_2"]].mean())
    print("\nSample cluster assignments (2019-20):")
    print(out[out["season"] == "2019-20"][["team", "cluster_id", "pca_1", "pca_2"]].sort_values("cluster_id").to_string())
