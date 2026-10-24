import argparse
from pathlib import Path

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from xg.features import add_features
from xg.prediction_log import log_path

ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "data" / "shots_curated.parquet"
MODEL = ROOT / "models" / "xg_model.joblib"
REFERENCE = ROOT / "models" / "reference.parquet"

INPUT_COLUMNS = [
    "x", "y", "shot_body_part", "shot_type", "shot_technique",
    "play_pattern", "shot_first_time", "under_pressure",
]
BOOLEANS = ["shot_first_time", "under_pressure"]
NUMERIC = ["x", "y", "distance", "angle", "xg"]
CATEGORICAL = ["shot_body_part", "shot_type", "shot_technique", "play_pattern", *BOOLEANS]


def _clean_bools(df: pd.DataFrame) -> pd.DataFrame:
    for c in BOOLEANS:
        df[c] = df[c].fillna(False).astype(bool)
    return df


def make_reference(n: int = 5000) -> None:
    """Muestra de los datos de entrenamiento, con el xG que predice el modelo."""
    df = pd.read_parquet(CURATED)
    ref = df[INPUT_COLUMNS].sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    ref = _clean_bools(ref)
    ref["xg"] = joblib.load(MODEL).predict_proba(ref)[:, 1]
    ref.to_parquet(REFERENCE)
    print(f"Referencia guardada: {len(ref)} filas en {REFERENCE}")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(_clean_bools(df[INPUT_COLUMNS + ["xg"]].copy()))
    for c in BOOLEANS:          # como texto, para tratarlas como categorías
        df[c] = df[c].astype(str)
    return df[NUMERIC + CATEGORICAL]


def make_report(out: Path, last: int | None) -> None:
    path = log_path()
    if not path.exists():
        raise SystemExit(f"No hay log en {path}. Manda predicciones a la API primero.")
    current = pd.read_json(path, lines=True)
    if last:
        current = current.tail(last)
    if len(current) < 30:
        raise SystemExit(f"Solo hay {len(current)} predicciones; junta al menos 30.")

    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    ref_ds = Dataset.from_pandas(_prepare(pd.read_parquet(REFERENCE)), data_definition=definition)
    cur_ds = Dataset.from_pandas(_prepare(current), data_definition=definition)

    result = Report([DataDriftPreset()]).run(cur_ds, ref_ds)   # primero lo actual, luego la referencia
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out))
    print(f"Reporte con {len(current)} predicciones guardado en {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reference")
    rep = sub.add_parser("report")
    rep.add_argument("--out", type=Path, default=Path("reports/drift.html"))
    rep.add_argument("--last", type=int, default=None, help="usar solo las últimas N predicciones")
    args = parser.parse_args()
    make_reference() if args.cmd == "reference" else make_report(args.out, args.last)