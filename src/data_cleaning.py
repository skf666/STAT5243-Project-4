"""Data cleaning + multi-source merge for EPL match data.

Builds the canonical match-level table by joining:
- football-data.co.uk (results, basic stats, betting odds) — primary
- ClubElo (per-team daily Elo rating) — strength enrichment
- FBref (xG, possession) — when available (2017-18+)
- BBC / Guardian NLP-derived features — when available

Output: `data/cleaned/matches.parquet` with one row per match and a
`source_coverage` column listing which enrichment sources contributed.

Reused from Project 2 with credits: pure-function loaders + scaler/encoder
helpers carried over verbatim. New here: `merge_match_sources()` and
`build_interim_matches()`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
CLEANED_DIR = Path("data/cleaned")


# ---------------------------------------------------------------------------
# Pure loaders (carried over from Project 2 / data_cleaning.py — credit team)
# ---------------------------------------------------------------------------


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(filepath, **kwargs)


def load_parquet(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def get_overview(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_col": df.isna().sum().to_dict(),
        "missing_pct_per_col": (df.isna().mean() * 100).round(2).to_dict(),
    }


# ---------------------------------------------------------------------------
# Multi-source merge (NEW for Project 4)
# ---------------------------------------------------------------------------


def attach_clubelo(matches: pd.DataFrame, elo_long: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match Elo rating to each (home_team, away_team) using `match_date`.

    `elo_long` has columns: team, From (datetime), To (datetime), Elo, Level.
    For each match row we look up the Elo interval that contains
    `match_date - 1 day` (so the rating is strictly pre-kickoff).
    """
    if elo_long.empty:
        matches["home_elo"] = np.nan
        matches["away_elo"] = np.nan
        return matches

    elo = elo_long[["team", "From", "To", "Elo"]].copy()
    elo["From"] = pd.to_datetime(elo["From"])
    elo["To"] = pd.to_datetime(elo["To"])
    elo = elo.rename(columns={"Elo": "elo"})

    def _lookup(team: str, date: pd.Timestamp) -> float:
        if pd.isna(date) or not isinstance(team, str):
            return np.nan
        target = date - pd.Timedelta(days=1)
        sub = elo[(elo["team"] == team) & (elo["From"] <= target) & (elo["To"] >= target)]
        if sub.empty:
            # fallback: most recent rating before target
            past = elo[(elo["team"] == team) & (elo["To"] < target)]
            return past["elo"].iloc[-1] if not past.empty else np.nan
        return sub["elo"].iloc[0]

    matches["home_elo"] = matches.apply(
        lambda r: _lookup(r["home_team"], r["match_date"]), axis=1
    )
    matches["away_elo"] = matches.apply(
        lambda r: _lookup(r["away_team"], r["match_date"]), axis=1
    )
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    return matches


def attach_nlp_features(
    matches: pd.DataFrame, nlp_features: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Left-join NLP-derived per-match features (sentiment, event tags)."""
    if nlp_features is None or nlp_features.empty:
        for col in (
            "home_report_sentiment",
            "away_report_sentiment",
            "controversy_flag",
            "red_card_mention_count",
            "penalty_mention_count",
            "report_length_words",
        ):
            matches[col] = np.nan
        matches["nlp_source"] = ""
        return matches
    return matches.merge(
        nlp_features,
        on=["match_date", "home_team", "away_team"],
        how="left",
    )


def attach_fbref_xg(matches: pd.DataFrame, fbref_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-join FBref xG by (date, home, away). Older seasons have NaN."""
    if fbref_df is None or fbref_df.empty:
        matches["home_xg"] = np.nan
        matches["away_xg"] = np.nan
        return matches
    cols_needed = ["date", "home_team", "away_team", "home_xg", "away_xg"]
    if not all(c in fbref_df.columns for c in cols_needed):
        return matches
    sub = fbref_df[cols_needed].rename(columns={"date": "match_date"})
    return matches.merge(sub, on=["match_date", "home_team", "away_team"], how="left")


def add_source_coverage(matches: pd.DataFrame) -> pd.DataFrame:
    """Add a string column listing which sources contributed to each row."""
    parts = []
    for _, row in matches.iterrows():
        sources = ["footballdata"]  # always present (primary key source)
        if pd.notna(row.get("home_elo")):
            sources.append("clubelo")
        if pd.notna(row.get("home_xg")):
            sources.append("fbref")
        if pd.notna(row.get("home_report_sentiment")):
            sources.append("bbc_or_guardian")
        parts.append(",".join(sources))
    matches = matches.copy()
    matches["source_coverage"] = parts
    return matches


def build_interim_matches(
    fd_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    fbref_df: Optional[pd.DataFrame] = None,
    nlp_df: Optional[pd.DataFrame] = None,
    *,
    out_path: str | Path = CLEANED_DIR / "matches.parquet",
) -> pd.DataFrame:
    """Run the full multi-source merge and persist."""
    LOG.info("Building interim matches from %d football-data rows", len(fd_df))
    matches = fd_df.copy()
    # Drop rows missing essential targets (rare).
    matches = matches.dropna(subset=["home_team", "away_team", "result"]).copy()
    matches = attach_clubelo(matches, elo_df)
    if fbref_df is not None:
        matches = attach_fbref_xg(matches, fbref_df)
    if nlp_df is not None:
        matches = attach_nlp_features(matches, nlp_df)
    matches = add_source_coverage(matches)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_path, index=False)
    LOG.info("Wrote %d matches to %s", len(matches), out_path)
    return matches


def quality_report(matches: pd.DataFrame) -> dict:
    """Per-column null rate, plus per-source coverage breakdown."""
    out = {
        "n_matches": len(matches),
        "n_seasons": matches["season"].nunique() if "season" in matches.columns else 0,
        "null_rate": (matches.isna().mean() * 100).round(2).to_dict(),
        "source_coverage": matches["source_coverage"].value_counts().to_dict()
        if "source_coverage" in matches.columns
        else {},
        "result_dist": matches["result"].value_counts(dropna=False).to_dict(),
    }
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from src.scrape.football_data_uk import fetch_seasons, all_target_seasons
    from src.scrape.club_elo import fetch_all_target_clubs

    fd = fetch_seasons(all_target_seasons())
    elo = fetch_all_target_clubs()
    matches = build_interim_matches(fd, elo)
    import pprint

    pprint.pp(quality_report(matches))
