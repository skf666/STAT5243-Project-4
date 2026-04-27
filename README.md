# STAT 5243 Project 4 — End-to-End ML for English Premier League Match Prediction

> **Status:** scaffold in progress. Predict the 2020-21 EPL season using 2000-01 through 2019-20 as training data (≈8,000 matches, 21 seasons). Multi-source self-collected data, including unstructured BBC match-report HTML processed via NLP into structured features.

**GitHub:** <https://github.com/ZemingLiang/STAT5243-Project-4>

**Group Members (TBD — confirm with team):**
- Zeming Liang (`zl3688`) — owner / end-to-end coordinator
- _yh3945-cmd_ — _confirm_ (invited via `yh3945@columbia.edu`)
- _xiyingchen_ — Xiying (Elina) Chen, MA Statistics @ Columbia (invited via `xiyingchen1@gmail.com`)

**Deployed App:** _TBD — Posit Connect Cloud_

---

## What this project does

End-to-end machine-learning pipeline that predicts English Premier League (EPL) match outcomes for the 2020-21 season from 21 prior seasons of self-collected data. Four interlinked predictive targets:

1. **Match outcome** — Home win / Draw / Away win (3-class classification)
2. **Exact score** — bivariate Poisson / Dixon-Coles regression on (home_goals, away_goals)
3. **Final season standings** — Monte Carlo simulation (10k iterations) of the 380-match season
4. **Beat the bookmaker** — log-loss / Brier / ROI vs Pinnacle closing odds with Kelly-fractional staking

Designed to score Advanced [10pt] on every rubric criterion of STAT 5243's final-project rubric.

---

## Data sources (all self-collected — no Kaggle)

**Structured / semi-structured:**
- **football-data.co.uk** — match results, half-time scores, betting odds (per-season CSV download, 2000-01..2020-21)
- **FBref / Sports Reference** — advanced stats including xG (2017-18+), via the `soccerdata` Python wrapper (3-second-rate-limited polite scrape)
- **ClubElo (clubelo.com)** — historical Elo ratings via per-club CSV API
- **Wikipedia** — manager tenure, promoted/relegated team metadata, season recap tables (`pandas.read_html`)

**Unstructured (the showcase of the cleaning pipeline):**
- **BBC Sport match reports** — ~7,600 HTML pages scraped, parsed via `BeautifulSoup`, then NLP pipeline extracts:
  - Player & manager entities (`spaCy en_core_web_sm` NER)
  - Per-team sentiment scores (VADER on per-team sentence subsets)
  - Key-event tags (red card / penalty / VAR / controversy / injury — regex classifier)
  - Aggregated to ~12 derived columns per match
- **Optional:** Premier League official press conference transcripts (best effort)

---

## Dataset files in this repo

To satisfy the deliverable requirement for both raw and processed data, this repository now includes:

- `data/raw/football_data_uk/*.parquet` — per-season raw match tables from football-data.co.uk
- `data/raw/club_elo/*.parquet` — per-club historical Elo snapshots from ClubElo
- `data/raw/wikipedia/*` — per-season recap tables and recap text artifacts
- `data/interim/matches.parquet` — source-harmonized match-level table
- `data/interim/matches.csv` — CSV export of the cleaned/interim match-level table
- `data/processed/matches.parquet` — final model-ready matrix
- `data/processed/matches.csv` — CSV export of the processed matrix for grading convenience

If you re-run the pipeline, these files are regenerated in place.

---

## Tech stack

- **App / UI:** Shiny for Python + shinywidgets + shinyswatch (Lux Bootstrap theme)
- **Plotting:** Plotly (with the Project-2 `app_theme` template for visual consistency)
- **Data:** pandas, numpy, pyarrow (parquet)
- **Scraping:** requests, BeautifulSoup, lxml, soccerdata, wikipedia-api
- **NLP:** spaCy (en_core_web_sm), vaderSentiment
- **Modeling:** scikit-learn, xgboost, lightgbm, statsmodels (Poisson GLM), optuna
- **Reporting:** pandoc + xelatex
- **Deployment:** Posit Connect Cloud
- **Testing:** unittest

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run model selection only (writes results/model_selection.csv and results/best_model_selection.json)
python -m src.model_selection --models all

# 3. Run the one-command pipeline:
#    feature engineering -> model selection -> final training
python -m src.pipeline

# 4. (Optional) Re-run only selection and training
python -m src.pipeline --skip-feature-engineering

# 5. Run the Shiny app
shiny run app.py

# 6. Generate the final PDF report
python -m src.fill_report --template REPORT.md --metrics results/leaderboard.csv --out REPORT.filled.md
pandoc REPORT.filled.md -o report.pdf --pdf-engine=xelatex --toc --toc-depth=2 \
       -V mainfont="Helvetica Neue" -V monofont=Menlo

# 7. Run smoke + leakage tests
python -m unittest tests
```

---

## Model selection workflow

`src/model_selection.py` runs time-aware (season-based expanding-window) cross-validation and compares hyperparameter candidates for:

- `logistic`
- `random_forest`
- `xgboost`
- `lightgbm`

Outputs:

- `results/model_selection.csv` — full candidate leaderboard
- `results/best_model_selection.json` — the selected best candidate

`src/train.py` automatically reads `results/best_model_selection.json` (if present) and applies those parameters to the matching model family during final training.

Use `python -m src.train --ignore-best-selection` to force default model parameters.

---

## Repository structure

```
STAT5243-Project-4/
├── README.md                    # this file
├── REPORT.md                    # report template with {placeholder} tokens
├── REPORT.filled.md             # pipeline-rendered report with numbers filled
├── report.pdf                   # final pandoc PDF (deliverable)
├── requirements.txt             # pinned deps
├── .github/workflows/tests.yml  # CI: unittest on push + PR
├── data/
│   ├── raw/                     # untouched scrapes (gitignored except small fixtures)
│   ├── interim/                 # source-harmonized, pre-feature-engineering
│   └── processed/               # final modeling matrices
├── src/
│   ├── scrape/                  # scrapers per data source
│   ├── nlp/                     # unstructured → structured NLP pipeline
│   ├── models/                  # baseline, logistic, RF, XGB, LightGBM, Dixon-Coles, stacked
│   ├── data_cleaning.py         # extended from Project 2
│   ├── feature_engineering.py   # extended from Project 2
│   ├── eda.py                   # extended from Project 2
│   ├── temporal_features.py     # leakage-safe rolling stats, Elo, rest, derbies
│   ├── unsupervised.py          # PCA, K-means, t-SNE/UMAP for team-style space
│   ├── team_harmonize.py        # canonical team-name mapping across sources
│   ├── evaluate.py              # full metrics suite + calibration + confusion
│   ├── betting.py               # closing-odds-implied probs + Kelly + ROI sim
│   ├── season_sim.py            # Monte Carlo 2020-21 standings (10k iter)
│   ├── model_selection.py       # time-aware CV + hyperparameter selection
│   ├── train.py                 # walk-forward CV + Optuna tuner CLI
│   ├── pipeline.py              # one-command FE -> selection -> train runner
│   └── fill_report.py           # populate REPORT.md placeholders from results/
├── notebooks/                   # exploratory (not the final deliverable)
├── app.py                       # Shiny app entrypoint (six tabs)
├── figures/                     # PNG / HTML exports referenced by REPORT.md
├── tests.py                     # integration + leakage + NLP fixture tests
└── test_data/                   # tiny fixtures for unit tests
```

---

## Critical correctness discipline

**No temporal leakage.** Football match prediction has a notorious leakage trap: random k-fold cross-validation lets future data inform past predictions. Our discipline:

- **Walk-forward (expanding-window) CV** — train on seasons 1..N, validate on N+1.
- **Strict `as_of` cutoff** on every rolling feature — only matches with kickoff `<` current match contribute.
- **Firewalled 2020-21 holdout** — features computed cleanly without ever touching 2020-21 results.
- **Unit-tested** by `tests.test_no_temporal_leakage`.

**Covid distribution shift.** The 2020-21 season was played behind closed doors for most matches, reducing home advantage. Documented as a "challenges faced" section; explicitly demonstrated by the home-win-rate-by-season plot in EDA.

---

## Team contributions (TBD — fill once roles are confirmed)

| Member | Contribution |
|---|---|
| Zeming Liang | TBD |
| Xiying Chen| TBD |
| _Collaborator 2_ | TBD |

---

## License

Academic project — code MIT, data subject to upstream sources' terms.
