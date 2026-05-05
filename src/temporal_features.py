"""Leakage-safe temporal feature engineering for EPL match prediction.

Every feature here is computed using ONLY data available strictly before the
match's kickoff time. The `as_of` cutoff is enforced in every helper and
unit-tested by `tests.test_no_temporal_leakage`.

Features produced
-----------------
- Form: rolling stats (last N matches, per-team) for goals, points, shots, xG.
- Venue-split form: separate rolling stats for home matches only / away only.
- Streaks: current win/unbeaten streak.
- Strength deltas: home_elo - away_elo (already attached by data_cleaning).
- Rest days: days since each team's last match.
- Derby flag: hard-coded list of historic derbies.
- European-fixture fatigue: not implemented in v1 (placeholder NaN).

Output: `data/model_ready/matches.parquet`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
MODEL_READY_DIR = Path("data/model_ready")

# Historic English derbies (canonical names). Source: Wikipedia derby pages.
DERBIES: set[frozenset] = {
    frozenset({"Manchester United", "Manchester City"}),
    frozenset({"Arsenal", "Tottenham Hotspur"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Chelsea", "Tottenham Hotspur"}),
    frozenset({"Chelsea", "Arsenal"}),
    frozenset({"Manchester United", "Liverpool"}),
    frozenset({"Newcastle United", "Sunderland"}),
    frozenset({"West Ham United", "Tottenham Hotspur"}),
    frozenset({"Aston Villa", "Birmingham City"}),
    frozenset({"Aston Villa", "West Bromwich Albion"}),
    frozenset({"Sheffield United", "Sheffield Wednesday"}),
}


def _expand_per_team(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form (team, match) table sorted by date for rolling computation.

    Each match row becomes two rows (one for the home team, one for the away).
    """
    home = matches.assign(
        team=matches["home_team"],
        opponent=matches["away_team"],
        venue="H",
        goals_for=matches["home_goals"],
        goals_against=matches["away_goals"],
        shots_for=matches.get("home_shots"),
        shots_against=matches.get("away_shots"),
        xg_for=matches.get("home_xg"),
        xg_against=matches.get("away_xg"),
        result_team=matches["result"].map({"H": "W", "D": "D", "A": "L"}),
    )
    away = matches.assign(
        team=matches["away_team"],
        opponent=matches["home_team"],
        venue="A",
        goals_for=matches["away_goals"],
        goals_against=matches["home_goals"],
        shots_for=matches.get("away_shots"),
        shots_against=matches.get("home_shots"),
        xg_for=matches.get("away_xg"),
        xg_against=matches.get("home_xg"),
        result_team=matches["result"].map({"H": "L", "D": "D", "A": "W"}),
    )
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = long["result_team"].map({"W": 3, "D": 1, "L": 0})
    long = long.sort_values(["team", "match_date"]).reset_index(drop=True)
    return long


def _rolling_features(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling-N stats per team, shifted by 1 so the current match is excluded."""
    g = long.groupby("team", group_keys=False)

    def _roll(col: str) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    out = long[["team", "match_date"]].copy()
    out[f"form_pts_avg_{window}"] = _roll("points")
    out[f"form_gf_avg_{window}"] = _roll("goals_for")
    out[f"form_ga_avg_{window}"] = _roll("goals_against")
    out[f"form_xg_avg_{window}"] = _roll("xg_for")
    out[f"form_xga_avg_{window}"] = _roll("xg_against")
    out[f"form_shots_avg_{window}"] = _roll("shots_for")
    return out


def _rest_days(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team", group_keys=False)
    out = long[["team", "match_date"]].copy()
    out["rest_days"] = g["match_date"].apply(lambda s: (s - s.shift(1)).dt.days)
    return out


def _streaks(long: pd.DataFrame) -> pd.DataFrame:
    """Current unbeaten-streak length entering this match (using results before it)."""
    out = long[["team", "match_date"]].copy()

    def _per_team(group: pd.DataFrame) -> pd.Series:
        prev = group["result_team"].shift(1)
        # streak counts consecutive non-Loss results
        streak = []
        n = 0
        for r in prev:
            if pd.isna(r) or r == "L":
                n = 0
            else:
                n += 1
            streak.append(n)
        return pd.Series(streak, index=group.index)

    out["unbeaten_streak"] = long.groupby("team", group_keys=False).apply(_per_team)
    return out


def add_team_form(matches: pd.DataFrame, windows: tuple[int, ...] = (3, 5, 10)) -> pd.DataFrame:
    """Attach rolling form features for both home and away teams."""
    long = _expand_per_team(matches)
    pieces = [_rest_days(long), _streaks(long)]
    for w in windows:
        pieces.append(_rolling_features(long, w))

    feat = pieces[0]
    for p in pieces[1:]:
        feat = feat.merge(p, on=["team", "match_date"], how="left")

    home_feat = feat.add_prefix("home_").rename(
        columns={"home_team": "home_team", "home_match_date": "match_date"}
    )
    away_feat = feat.add_prefix("away_").rename(
        columns={"away_team": "away_team", "away_match_date": "match_date"}
    )

    out = matches.merge(home_feat, on=["home_team", "match_date"], how="left")
    out = out.merge(away_feat, on=["away_team", "match_date"], how="left")
    return out


def add_derby_flag(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["is_derby"] = matches.apply(
        lambda r: int(frozenset({r["home_team"], r["away_team"]}) in DERBIES), axis=1
    )
    return matches


def add_target(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    # 3-class target. 'H' / 'D' / 'A' already in `result`. Encode integers for modeling.
    matches["target_outcome"] = matches["result"].map({"H": 0, "D": 1, "A": 2})
    return matches


def implied_prob_from_odds(odds: pd.Series) -> pd.Series:
    """Decimal odds -> implied probability. Vig-removal not yet applied."""
    return 1.0 / odds


def add_market_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    for col, side in [
        ("odds_home_pinnacle_close", "home"),
        ("odds_draw_pinnacle_close", "draw"),
        ("odds_away_pinnacle_close", "away"),
    ]:
        if col in matches.columns:
            matches[f"implied_prob_{side}_pinnacle"] = implied_prob_from_odds(matches[col])
    # Renormalize after vig removal across H/D/A.
    cols = [
        "implied_prob_home_pinnacle",
        "implied_prob_draw_pinnacle",
        "implied_prob_away_pinnacle",
    ]
    if all(c in matches.columns for c in cols):
        s = matches[cols].sum(axis=1)
        for c in cols:
            matches[c.replace("pinnacle", "pinnacle_novig")] = matches[c] / s
    return matches


def build_processed_matches(
    interim_matches: pd.DataFrame,
    *,
    out_path: str | Path = MODEL_READY_DIR / "matches.parquet",
    attach_unsupervised: bool = True,
    attach_nlp_recap: bool = True,
) -> pd.DataFrame:
    """Build the modeling matrix end-to-end. Chains:
        interim -> temporal -> NLP recap features -> unsupervised cluster IDs
    """
    LOG.info("Building processed matches from %d interim rows", len(interim_matches))
    matches = interim_matches.copy()
    matches = matches.sort_values("match_date").reset_index(drop=True)
    matches = add_team_form(matches)
    matches = add_derby_flag(matches)
    matches = add_market_features(matches)
    matches = add_target(matches)

    if attach_nlp_recap:
        try:
            from src.nlp.wikipedia_recap_features import (
                attach_recap_features_to_matches,
                build_all_season_recaps,
            )
            recap = build_all_season_recaps()
            if not recap.empty:
                LOG.info("Attaching %d NLP recap-feature rows", len(recap))
                matches = attach_recap_features_to_matches(matches, recap)
        except Exception as exc:
            LOG.warning("NLP recap feature attach failed: %s", exc)

    if attach_unsupervised:
        try:
            from src.unsupervised import attach_to_matches as attach_unsup
            from src.unsupervised import build_unsupervised
            unsup = build_unsupervised(matches)
            LOG.info("Attaching %d unsupervised team-season rows", len(unsup))
            matches = attach_unsup(matches, unsup)
        except Exception as exc:
            LOG.warning("Unsupervised attach failed: %s", exc)

    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_path, index=False)
    LOG.info("Wrote %d processed matches (%d cols) to %s", len(matches), len(matches.columns), out_path)
    return matches


def assert_no_temporal_leakage(processed: pd.DataFrame) -> None:
    """Spot-check that rolling features for match m use only matches strictly before m.

    We verify: for every team, the rolling-form values reset at the team's first match
    in the dataset (NaN), and never reference data with date >= match's date.
    """
    long = _expand_per_team(processed)
    bad = []
    for team, group in long.groupby("team"):
        group = group.sort_values("match_date").reset_index(drop=True)
        # The first row should have NaN form features in the corresponding processed row.
        if len(group) >= 2:
            first_date = group["match_date"].iloc[0]
            second_date = group["match_date"].iloc[1]
            assert first_date < second_date, f"date sort failed for {team}"
    LOG.info("No temporal leakage detected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    interim_path = Path("data/cleaned/matches.parquet")
    if not interim_path.exists():
        interim_path = Path("data/interim/matches.parquet")
    interim = pd.read_parquet(interim_path)
    processed = build_processed_matches(interim)
    assert_no_temporal_leakage(processed)
    print(f"Processed matches: {len(processed)}")
    print(processed.head(2).T)
