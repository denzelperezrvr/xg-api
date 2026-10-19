from pathlib import Path

import pandas as pd
from statsbombpy import sb

DATA_DIR = Path("data")

COMPETITIONS = [(9, 281), (43, 3), (43, 106), (55, 43), (55, 282)]

def download_shots() -> pd.DataFrame:
    frames = []
    for comp_id, season_id in zip(*sb.competitions()[["competition_id", "season_id"]].values.T):
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        for match_id in matches["match_id"]:
            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].copy()
            shots["match_id"] = match_id
            frames.append(shots)
            print(f"match {match_id}: {len(shots)} tiros")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    shots = download_shots()
    shots.to_parquet(DATA_DIR / "shots_raw.parquet")
    print(f"Total: {len(shots)} tiros guardados")
