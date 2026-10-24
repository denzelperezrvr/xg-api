import argparse
import random

import httpx
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
parser.add_argument("--n", type=int, default=500)
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()

ref = pd.read_parquet("models/reference.parquet").drop(columns=["xg"])
shots = ref.sample(n=args.n, replace=True, random_state=1).to_dict(orient="records")

if args.mode == "drift":
    # Tiros lejanos, muchos de cabeza y casi siempre bajo presión
    for s in shots:
        s["x"] = random.uniform(60, 95)
        s["y"] = random.uniform(5, 75)
        s["under_pressure"] = random.random() < 0.7
        if random.random() < 0.5:
            s["shot_body_part"] = "Head"

with httpx.Client(timeout=30) as client:
    for i in range(0, len(shots), 100):
        r = client.post(f"{args.url}/predict/batch", json={"shots": shots[i : i + 100]})
        r.raise_for_status()
print(f"Enviados {len(shots)} tiros en modo {args.mode}")