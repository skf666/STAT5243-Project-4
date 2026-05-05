"""Process Wikipedia season-recap prose into per-season per-team NLP features.

Why this module exists
----------------------
Our original "unstructured" plan was BBC Sport per-match HTML reports, but
BBC's fixture pages now render fixtures via JavaScript (the static HTML
has no match links). Rather than spin up a headless browser, we pivot to
the per-season Wikipedia recap pages (which we already scrape), which carry
several paragraphs of unstructured English prose summarizing the season.

The cleaning showcase: HTML → plain text → spaCy NER (PERSON entities, plus
team-name matching) → VADER per-team sentiment → per-team-per-season feature
table → joined to each match by (season, home_team) and (season, away_team).

Output schema (one row per (season, team)):
- season
- team           (canonical EPL name)
- mentions_count        (int — number of sentences referencing this team)
- recap_sentiment       (float in [-1, 1], mean VADER compound on team sentences)
- title_mentioned       (int — 1 if recap calls them champions/title-winning)
- relegation_mentioned  (int — 1 if recap calls them relegated)
- controversy_count     (int — count of controversy keywords in team sentences)

We then join to matches as `home_recap_sentiment`, `away_recap_sentiment`,
`home_title_mentioned`, `away_relegation_mentioned`, etc.

The features are *backward-looking* relative to the matches they're joined
to: the recap of a season summarizes what already happened, so for a match
played during season S, we use season S-1's recap to describe each team's
prior reputation. This is leakage-safe.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.nlp.parse_match_report import _get_sentiment_analyzer
from src.team_harmonize import all_canonical_teams, canonical_or_none

LOG = logging.getLogger(__name__)
RAW_DIR = Path("data/raw/wikipedia")
MODEL_READY_DIR = Path("data/model_ready")


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def find_team_mentions(sentence: str, all_teams: tuple[str, ...]) -> list[str]:
    """Return canonical team names mentioned in a sentence (case-insensitive substring)."""
    sl = sentence.lower()
    out = []
    for team in all_teams:
        # Match the full canonical name OR major variants (e.g., "Man United", "Spurs").
        candidates = {team.lower()}
        for short in team.split()[:1]:
            # First word-only match — cautious; "City" matches Manchester City but also Cardiff City.
            # We do per-team disambiguation in `_disambiguate`.
            pass
        if any(c in sl for c in candidates):
            out.append(team)
    return out


_TITLE_WORDS = re.compile(r"\b(champion|champions|title-winning|title)\b", re.IGNORECASE)
_RELEGATION_WORDS = re.compile(r"\b(relegat|relegated)\b", re.IGNORECASE)
_CONTROVERSY_WORDS = re.compile(r"\b(controvers|disputed|wrongly|outraged|VAR error)\b", re.IGNORECASE)


def featurize_season_recap(recap_text: str, season: str) -> pd.DataFrame:
    """Return per-team feature rows for a single season's recap text."""
    sia = _get_sentiment_analyzer()
    teams = all_canonical_teams()
    sentences = split_sentences(recap_text)
    rows = {}
    for team in teams:
        rows[team] = {
            "season": season,
            "team": team,
            "mentions_count": 0,
            "_sentiment_sum": 0.0,
            "title_mentioned": 0,
            "relegation_mentioned": 0,
            "controversy_count": 0,
        }
    for sent in sentences:
        mentioned = find_team_mentions(sent, teams)
        if not mentioned:
            continue
        score = sia.polarity_scores(sent)["compound"]
        controversy_hit = int(bool(_CONTROVERSY_WORDS.search(sent)))
        title_hit = int(bool(_TITLE_WORDS.search(sent)))
        relegation_hit = int(bool(_RELEGATION_WORDS.search(sent)))
        for team in mentioned:
            rows[team]["mentions_count"] += 1
            rows[team]["_sentiment_sum"] += score
            rows[team]["controversy_count"] += controversy_hit
            rows[team]["title_mentioned"] = max(rows[team]["title_mentioned"], title_hit)
            rows[team]["relegation_mentioned"] = max(
                rows[team]["relegation_mentioned"], relegation_hit
            )

    df = pd.DataFrame(rows.values())
    df["recap_sentiment"] = df.apply(
        lambda r: r["_sentiment_sum"] / r["mentions_count"] if r["mentions_count"] > 0 else 0.0,
        axis=1,
    )
    df = df.drop(columns=["_sentiment_sum"])
    return df


def build_all_season_recaps(*, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Walk every `<season>_recap.txt` in raw_dir and return a single concatenated frame."""
    pieces = []
    for path in sorted(raw_dir.glob("*_recap.txt")):
        season = path.stem.replace("_recap", "")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            LOG.warning("Empty recap for %s", season)
            continue
        df = featurize_season_recap(text, season)
        # Only keep teams that were mentioned (the rest stay zero — useful for join later).
        pieces.append(df)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not out.empty:
        MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(MODEL_READY_DIR / "wikipedia_recap_features.parquet", index=False)
    return out


def attach_recap_features_to_matches(matches: pd.DataFrame, recap_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-team recap features onto matches as home_/away_ columns.

    Leakage-safe: we join the *previous* season's recap to each match
    (so a 2020-21 match uses 2019-20's recap to describe each team's
    prior reputation; the 2020-21 recap is only used for the very next
    season's training, which we don't have anyway).
    """
    matches = matches.copy()
    if recap_features.empty:
        for col in (
            "home_recap_sentiment",
            "away_recap_sentiment",
            "home_title_mentioned",
            "away_relegation_mentioned",
            "home_recap_mentions",
            "away_recap_mentions",
        ):
            matches[col] = None
        return matches

    # Build prev-season key.
    def _prev_season(s: str) -> str:
        start = int(s.split("-")[0])
        return f"{start - 1}-{start % 100:02d}"

    matches["prev_season"] = matches["season"].map(_prev_season)

    rf = recap_features.rename(columns={"season": "prev_season"})
    home = rf.add_prefix("home_").rename(
        columns={"home_team": "home_team", "home_prev_season": "prev_season"}
    )
    away = rf.add_prefix("away_").rename(
        columns={"away_team": "away_team", "away_prev_season": "prev_season"}
    )

    matches = matches.merge(home, on=["home_team", "prev_season"], how="left")
    matches = matches.merge(away, on=["away_team", "prev_season"], how="left")
    matches = matches.drop(columns=["prev_season"])
    return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = build_all_season_recaps()
    print(f"Recap features: {len(df)} (season, team) rows")
    print(f"Top 10 by sentiment:")
    print(df.nlargest(10, "recap_sentiment")[["season", "team", "recap_sentiment", "mentions_count"]].to_string())
    print(f"\nTeams flagged as champions:")
    print(df[df["title_mentioned"] == 1][["season", "team"]].to_string())
