"""StatsBomb open data (all free competitions) -> training_shots.parquet.

One row per shot, with the key-pass attributes joined on. Penalty shootouts (period 5) are dropped.
"""
import warnings, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from statsbombpy import sb
import requests_cache
requests_cache.uninstall_cache()  # statsbombpy caches every response in a tmp sqlite (filled /tmp with 4 GB)

warnings.filterwarnings("ignore")
OUT = "training_shots.parquet"


def match_shots(m):
    mid = m["match_id"]
    for _ in range(3):
        try:
            ev = sb.events(match_id=mid)
            break
        except Exception:
            ev = None
    if ev is None:
        return mid, None
    ev = ev.copy()
    for c in ["shot_key_pass_id", "pass_cross", "pass_through_ball", "shot_statsbomb_xg", "shot_type",
              "shot_body_part", "shot_outcome", "play_pattern", "location", "period", "id"]:
        if c not in ev:
            ev[c] = None
    shots = ev[(ev.type == "Shot") & (ev.period < 5)].copy()
    if shots.empty:
        return mid, shots
    kp = ev[ev.type == "Pass"].set_index("id")[["pass_cross", "pass_through_ball"]]
    kp = kp.rename(columns={"pass_cross": "kp_cross", "pass_through_ball": "kp_through"})
    shots = shots.join(kp, on="shot_key_pass_id")
    shots["match_id"] = mid
    shots["competition"] = m["competition"]
    shots["season"] = m["season"]
    keep = ["match_id", "competition", "season", "location", "shot_outcome", "shot_body_part", "shot_type",
            "play_pattern", "shot_key_pass_id", "kp_cross", "kp_through", "shot_statsbomb_xg", "under_pressure"]
    for c in keep:
        if c not in shots:
            shots[c] = None
    return mid, shots[keep]


if __name__ == "__main__":
    comps = sb.competitions()
    print("competition-seasons:", len(comps))
    ms = []
    for _, c in comps.iterrows():
        try:
            m = sb.matches(competition_id=c.competition_id, season_id=c.season_id)
        except Exception as e:
            print("skip", c.competition_name, c.season_name, e); continue
        m["competition"] = f"{c.competition_name} ({c.competition_gender})"
        m["season"] = c.season_name
        ms.append(m[["match_id", "competition", "season"]])
    ms = pd.concat(ms).drop_duplicates("match_id")
    print("matches:", len(ms))
    rows, failed = [], []
    with ThreadPoolExecutor(16) as ex:
        for i, (mid, s) in enumerate(ex.map(match_shots, ms.to_dict("records"))):
            if s is None: failed.append(mid)
            else: rows.append(s)
            if i % 200 == 0: print(i, flush=True)
    df = pd.concat(rows, ignore_index=True)
    print("failed matches:", len(failed), "shots:", len(df))
    df["x"] = df.location.map(lambda l: l[0] if isinstance(l, (list, tuple)) or hasattr(l, "__len__") else None)
    df["y"] = df.location.map(lambda l: l[1] if hasattr(l, "__len__") else None)
    df = df.drop(columns="location")
    for c in ["kp_cross", "kp_through", "under_pressure"]:
        df[c] = df[c].eq(True)
    df.to_pickle("training_shots.pkl")  # safety copy
    df.to_parquet(OUT)
    json.dump(failed, open("failed_matches.json", "w"))
    print(df.competition.value_counts().head(30))
