"""Time-aware model selection for EPL match outcome prediction.

Runs expanding-window cross-validation by season on the processed matches table,
searches a compact hyperparameter grid for each model family, and writes:

- results/model_selection.csv
- results/best_model_selection.json

Usage:
    python -m src.model_selection
    python -m src.model_selection --models logistic,xgboost
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.train import available_features

LOG = logging.getLogger(__name__)
MODEL_READY = Path("data/model_ready/matches.parquet")
LEGACY_MODEL_READY = Path("data/processed/matches.parquet")
RESULTS_DIR = Path("results")


def _make_logistic(features: list[str], params: dict):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=20260426,
        C=float(params["C"]),
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _make_random_forest(features: list[str], params: dict):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=20260426,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _make_xgboost(features: list[str], params: dict):
    import xgboost as xgb
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=20260426,
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _make_lightgbm(features: list[str], params: dict):
    import lightgbm as lgb
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), features)],
        remainder="drop",
    )
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=20260426,
        verbosity=-1,
        n_estimators=int(params["n_estimators"]),
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _param_grid_to_list(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _season_folds(df: pd.DataFrame) -> list[tuple[list[str], str]]:
    seasons = sorted(df["season"].dropna().unique())
    # Keep 2020-21 as final holdout, selection uses earlier validation seasons.
    valid_targets = [s for s in seasons if s not in {"2000-01", "2020-21"}]
    folds = []
    for val_season in valid_targets:
        train_seasons = [s for s in seasons if s < val_season and s != "2020-21"]
        if len(train_seasons) >= 3:
            folds.append((train_seasons, val_season))
    return folds


def _safe_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y_true, y_proba, labels=[0, 1, 2]))


def evaluate_candidate(
    df: pd.DataFrame,
    features: list[str],
    factory: Callable[[list[str], dict], object],
    params: dict,
) -> tuple[float, list[dict]]:
    folds = _season_folds(df)
    fold_rows = []
    losses = []

    for train_seasons, val_season in folds:
        tr = df[df["season"].isin(train_seasons)]
        va = df[df["season"] == val_season]
        if tr.empty or va.empty:
            continue
        model = factory(features, params)
        model.fit(tr[features], tr["target_outcome"].to_numpy())
        proba = model.predict_proba(va[features])
        loss = _safe_log_loss(va["target_outcome"].to_numpy(), proba)
        losses.append(loss)
        fold_rows.append(
            {
                "val_season": val_season,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "log_loss": float(loss),
            }
        )

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    return mean_loss, fold_rows


def run(models: list[str]) -> pd.DataFrame:
    data_path = MODEL_READY if MODEL_READY.exists() else LEGACY_MODEL_READY
    if not data_path.exists():
        raise FileNotFoundError(f"{MODEL_READY} not found. Run feature engineering first.")

    df = pd.read_parquet(data_path).sort_values("match_date").reset_index(drop=True)
    if "target_outcome" not in df.columns:
        raise ValueError("`target_outcome` missing from processed data.")

    features = available_features(df)
    LOG.info("Model selection using %d features", len(features))

    registry = {
        "logistic": (
            _make_logistic,
            {"C": [0.25, 1.0, 2.5]},
        ),
        "random_forest": (
            _make_random_forest,
            {"n_estimators": [300, 500], "max_depth": [8, 12], "min_samples_leaf": [3, 6]},
        ),
        "xgboost": (
            _make_xgboost,
            {
                "n_estimators": [350, 600],
                "max_depth": [4, 6],
                "learning_rate": [0.03, 0.07],
                "subsample": [0.8],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            _make_lightgbm,
            {
                "n_estimators": [350, 600],
                "num_leaves": [31, 63],
                "learning_rate": [0.03, 0.07],
                "subsample": [0.8],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }

    if "all" in models:
        selected = list(registry.keys())
    else:
        selected = [m for m in models if m in registry]
    if not selected:
        raise ValueError("No valid models selected. Use --models all or a comma list.")

    rows = []
    for name in selected:
        factory, grid = registry[name]
        candidates = _param_grid_to_list(grid)
        LOG.info("Selecting %s across %d candidates", name, len(candidates))
        for i, params in enumerate(candidates, start=1):
            mean_loss, fold_rows = evaluate_candidate(df, features, factory, params)
            rows.append(
                {
                    "model": name,
                    "candidate_idx": i,
                    "params": json.dumps(params, sort_keys=True),
                    "cv_log_loss_mean": mean_loss,
                    "n_folds": len(fold_rows),
                }
            )

    out = pd.DataFrame(rows).sort_values(["cv_log_loss_mean", "model"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    best = out.iloc[0].to_dict()
    (RESULTS_DIR / "best_model_selection.json").write_text(json.dumps(best, indent=2))
    LOG.info("Best model: %s with CV log-loss %.5f", best["model"], best["cv_log_loss_mean"])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Time-aware model selection on processed EPL data.")
    parser.add_argument("--models", default="all", help="all or comma list: logistic,random_forest,xgboost,lightgbm")
    args = parser.parse_args()

    table = run(args.models.split(","))
    print(table.head(15).to_string(index=False))
