import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from xg.features import BOOLEAN, CATEGORICAL, NUMERIC, add_features

DATA = Path("data/shots_curated.parquet")
MODEL_PATH = Path("models/xg_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def build_pipeline(model, numeric, boolean=(), categorical=()):
    transformers = [("num", StandardScaler(), list(numeric))]
    if boolean:
        transformers.append(("bool", "passthrough", list(boolean)))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical))
        )
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", ColumnTransformer(transformers)),
        ("model", model),
    ])


def evaluate(name, pipe, X, y, groups):
    cv = GroupKFold(n_splits=5)
    p = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return {
        "model": name,
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
    }


if __name__ == "__main__":
    df = pd.read_parquet(DATA)
    y = df["target"]
    groups = df["match_id"]
    X = df.drop(columns=["target", "id", "match_id"])

    prior = y.mean()
    results = [{
        "model": "prior (sin features)",
        "log_loss": log_loss(y, np.full(len(y), prior)),
        "brier": brier_score_loss(y, np.full(len(y), prior)),
        "roc_auc": 0.5,
    }]

    baseline = build_pipeline(LogisticRegression(max_iter=1000), ["distance", "angle"])
    results.append(evaluate("logreg distance+angle", baseline, X, y, groups))

    full_lr = build_pipeline(
        LogisticRegression(max_iter=1000), NUMERIC, BOOLEAN, CATEGORICAL
    )
    results.append(evaluate("logreg todas las features", full_lr, X, y, groups))

    xgb = build_pipeline(
        XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=5, random_state=42, eval_metric="logloss",
        ),
        NUMERIC, BOOLEAN, CATEGORICAL,
    )
    results.append(evaluate("xgboost", xgb, X, y, groups))

    table = pd.DataFrame(results).round(4)
    print(table.to_string(index=False))

    # Entrena el mejor con todos los datos y guárdalo
    best = table.iloc[1:].sort_values("log_loss").iloc[0]["model"]
    final = {"logreg distance+angle": baseline, "logreg todas las features": full_lr,
             "xgboost": xgb}[best]
    final.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "model": best,
        "trained_on": str(date.today()),
        "n_shots": len(df),
        "cv_results": table.to_dict(orient="records"),
    }, indent=2))
    print(f"Guardado: {best}")