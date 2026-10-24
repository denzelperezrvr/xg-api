import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("xg-api")
_lock = threading.Lock()


def log_path() -> Path:
    return Path(os.getenv("XG_LOG_PATH", "logs/predictions.jsonl"))


def log_predictions(inputs: list[dict], xgs: list[float], model_version: str) -> None:
    """Agrega una línea JSON por predicción. Nunca debe romper la API."""
    path = log_path()
    ts = datetime.now(UTC).isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, path.open("a", encoding="utf-8") as f:
            for inp, xg in zip(inputs, xgs):
                record = {"ts": ts, "model_version": model_version, **inp, "xg": xg}
                f.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("No se pudo escribir el log de predicciones")