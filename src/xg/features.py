import numpy as np
import pandas as pd

GOAL_X = 120
GOAL_Y = 40
GOAL_WIDTH = 8  # metros entre postes (y=36 a y=44)

NUMERIC = ["distance", "angle"]
BOOLEAN = ["shot_first_time", "under_pressure"]
CATEGORICAL = ["shot_body_part", "shot_type", "shot_technique"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dx = (GOAL_X - df["x"]).clip(lower=0)
    dy = GOAL_Y - df["y"]

    df["distance"] = np.hypot(dx, dy)
    # ángulo de visión de la portería: el que forman los dos postes desde el tiro
    df["angle"] = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    df["shot_first_time"] = df["shot_first_time"].fillna(False).astype(bool)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(bool)
    return df