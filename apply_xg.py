"""shots.csv -> shots_xg.csv (Sporting's shots only, per user decision) + per-match totals."""
import pickle, sys
import numpy as np, pandas as pd
from features import features_espn, Y_SCALE

def run(y_scale=Y_SCALE, write=True):
    art = pickle.load(open("xg_model.pkl", "rb"))
    d = pd.read_csv("shots.csv")
    print("shots.csv rows:", len(d))
    og = d.type.eq("own-goal"); print("own goals removed:", og.sum())
    d = d[~og]
    keep = d.team.eq("Sporting CP"); print("opponent shots not modelled (user choice):", (~keep).sum())
    d = d[keep].copy()
    bad = d.fx.eq(0); print("rows with fx==0 (missing coords) -> xg NaN:", bad.sum())
    F = features_espn(d, y_scale)
    xg = art["iso"].transform(art["model"].predict_proba(F[art["features"]])[:, 1])
    xg = np.where(F.is_penalty == 1, art["penalty_rate"], xg)
    xg = np.where(bad, np.nan, xg)
    d["xg"] = xg
    d["is_goal"] = d.type.isin(["goal", "goal---header", "penalty---scored"])
    d = pd.concat([d, F.drop(columns=[c for c in ["is_penalty"] if False])], axis=1)
    if write: d.to_csv("shots_xg.csv", index=False)
    return d

if __name__ == "__main__":
    d = run()
    print("xg NaN count:", d.xg.isna().sum(), "| penalties set to", round(d[d.is_penalty == 1].xg.mean(), 3))
    t = d.groupby(["date", "match"]).agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("is_goal", "sum")).round(2)
    print(t); t.to_csv("match_xg_totals.csv")
    print("TOTAL xG", round(d.xg.sum(), 2), "goals", int(d.is_goal.sum()))
