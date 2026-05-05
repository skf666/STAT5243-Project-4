"""Feature engineering entrypoint for EPL match prediction.

This module provides a stable API layer over `src.temporal_features` so the
project can run:

    python -m src.feature_engineering

It reads cleaned merged matches, builds leakage-safe features, and writes the
model-ready matrix.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.temporal_features import build_processed_matches

LOG = logging.getLogger(__name__)

CLEANED_DEFAULT = Path("data/cleaned/matches.parquet")
LEGACY_CLEANED_DEFAULT = Path("data/interim/matches.parquet")
MODEL_READY_DEFAULT = Path("data/model_ready/matches.parquet")


def engineer_features(
    interim_df: pd.DataFrame,
    *,
    include_unsupervised: bool = True,
    include_nlp_recap: bool = True,
    out_path: str | Path = MODEL_READY_DEFAULT,
) -> pd.DataFrame:
    """Build and return a model-ready feature matrix from interim matches."""
    df = interim_df.copy()
    if "match_date" not in df.columns:
        raise ValueError("Expected `match_date` in interim data.")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    # Add a few robust interaction features often useful for tree + linear models.
    if {"home_elo", "away_elo"}.issubset(df.columns):
        df["elo_ratio"] = df["home_elo"] / df["away_elo"].replace(0, pd.NA)
    if {"home_xg", "away_xg"}.issubset(df.columns):
        df["xg_diff_raw"] = df["home_xg"] - df["away_xg"]

    processed = build_processed_matches(
        df,
        out_path=out_path,
        attach_unsupervised=include_unsupervised,
        attach_nlp_recap=include_nlp_recap,
    )
    return processed


def run(
    *,
    interim_path: str | Path = CLEANED_DEFAULT,
    out_path: str | Path = MODEL_READY_DEFAULT,
    include_unsupervised: bool = True,
    include_nlp_recap: bool = True,
) -> pd.DataFrame:
    """Load interim data, run feature engineering, persist output, return DataFrame."""
    interim_path = Path(interim_path)
    if not interim_path.exists() and interim_path == CLEANED_DEFAULT and LEGACY_CLEANED_DEFAULT.exists():
        interim_path = LEGACY_CLEANED_DEFAULT
    if not interim_path.exists():
        raise FileNotFoundError(
            f"{interim_path} not found. Run `python -m src.data_cleaning` first."
        )

    LOG.info("Loading cleaned matches from %s", interim_path)
    interim_df = pd.read_parquet(interim_path)
    processed = engineer_features(
        interim_df,
        include_unsupervised=include_unsupervised,
        include_nlp_recap=include_nlp_recap,
        out_path=out_path,
    )
    LOG.info("Feature engineering complete: %d rows, %d columns", len(processed), len(processed.columns))
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Build feature-engineered EPL match matrix.")
    parser.add_argument("--interim-path", default=str(INTERIM_DEFAULT))
    parser.add_argument("--out-path", default=str(PROCESSED_DEFAULT))
    parser.add_argument(
        "--no-unsupervised",
        action="store_true",
        help="Skip PCA/cluster team-style features.",
    )
    parser.add_argument(
        "--no-nlp-recap",
        action="store_true",
        help="Skip Wikipedia recap NLP aggregate features.",
    )
    args = parser.parse_args()

    out = run(
        interim_path=args.interim_path,
        out_path=args.out_path,
        include_unsupervised=not args.no_unsupervised,
        include_nlp_recap=not args.no_nlp_recap,
    )
    print(f"Processed matches: {len(out)} rows, {len(out.columns)} columns")
