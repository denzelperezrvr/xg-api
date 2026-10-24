import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("xg-api")

ROOT = Path(__file__).resolve().parents[2]
model = joblib.load(ROOT / "models" / "xg_model.joblib")
meta = json.loads((ROOT / "models" / "metrics.json").read_text())
MODEL_VERSION = f"{meta['model']}-{meta['trained_on']}"

BodyPart = Literal["Head", "Left Foot", "Right Foot", "Other"]
ShotType = Literal["Open Play", "Free Kick", "Corner", "Kick Off"]
Technique = Literal[
    "Normal", "Half Volley", "Volley", "Lob", "Diving Header", "Overhead Kick", "Backheel"
]
PlayPattern = Literal[
    "Regular Play", "From Corner", "From Counter", "From Free Kick",
    "From Goal Kick", "From Keeper", "From Kick Off", "From Throw In", "Other",
]


class Shot(BaseModel):
    # Coordenadas de StatsBomb: campo de 120x80, portería rival en x=120
    x: float = Field(ge=0, le=120, description="Posición a lo largo del campo")
    y: float = Field(ge=0, le=80, description="Posición a lo ancho del campo")
    shot_body_part: BodyPart
    shot_type: ShotType = "Open Play"
    shot_technique: Technique = "Normal"
    play_pattern: PlayPattern = "Regular Play"
    shot_first_time: bool = False
    under_pressure: bool = False


class BatchRequest(BaseModel):
    shots: list[Shot] = Field(min_length=1, max_length=1000)


app = FastAPI(title="xG API", version="1.0")


def predict_xg(shots: list[Shot]) -> list[float]:
    df = pd.DataFrame([s.model_dump() for s in shots])
    probs = model.predict_proba(df)[:, 1]
    return [round(float(p), 4) for p in probs]

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(shot: Shot):
    start = time.perf_counter()
    xg = predict_xg([shot])[0]
    ms = (time.perf_counter() - start) * 1000
    logger.info("predict xg=%.4f latency_ms=%.1f input=%s", xg, ms, shot.model_dump())
    return {"xg": xg, "model_version": MODEL_VERSION}


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    start = time.perf_counter()
    xgs = predict_xg(req.shots)
    ms = (time.perf_counter() - start) * 1000
    logger.info("predict_batch n=%d latency_ms=%.1f", len(xgs), ms)
    return {"xg": xgs, "model_version": MODEL_VERSION}