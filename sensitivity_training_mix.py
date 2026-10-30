"""Does the training mix matter? Retrain on men's elite club+international only, re-apply, compare to FotMob.

Writes nothing over the shipped artefacts; results go to metrics_mensonly.json.
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import MODEL_FEATURES, features_espn, features_statsbomb

DROP = ["Indian Super league", "FIFA U20 World Cup", "North American League", "Major League Soccer"]

def pool(raw, mens_only):
    if mens_only:
        raw = raw[raw.competition.str.contains(r"\(male\)")]
        raw = raw[~raw.competition.str.replace(r" \(male\)", "", regex=True).isin(DROP)]
    raw = raw[raw.x.notna()]
    pen_rate = float(raw[raw.shot_type.eq("Penalty")].shot_outcome.eq("Goal").mean())
    raw = raw[~raw.shot_type.isin(["Penalty", "Free Kick"])]
    raw = raw[(raw.x < 120.01) & (raw.x >= 60)]
    return raw.reset_index(drop=True), pen_rate

def fit(raw):
    X, y, g = features_statsbomb(raw), raw.shot_outcome.eq("Goal").astype(int).values, raw.match_id.values
    rest, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, g))
    tr_i, ca_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=1).split(X.iloc[rest], y[rest], g[rest]))
    tr, ca = rest[tr_i], rest[ca_i]
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                       l2_regularization=1.0, min_samples_leaf=100, random_state=0)
    m.fit(X.iloc[tr][MODEL_FEATURES], y[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.99)
    iso.fit(m.predict_proba(X.iloc[ca][MODEL_FEATURES])[:, 1], y[ca])
    p = iso.transform(m.predict_proba(X.iloc[te][MODEL_FEATURES])[:, 1])
    sb = raw.shot_statsbomb_xg.astype(float).values[te]
    ok = ~np.isnan(sb)
    return m, iso, dict(n_train=len(tr), n_test=len(te), base_rate=float(y[tr].mean()),
                        log_loss=float(log_loss(y[te], p)), roc_auc=float(roc_auc_score(y[te], p)),
                        test_goals=int(y[te].sum()), test_xg=float(p.sum()),
                        r_vs_statsbomb=float(np.corrcoef(p[ok], sb[ok])[0, 1]))

raw_all = pd.read_parquet("training_shots.parquet")
espn = pd.read_csv("shots.csv")
espn = espn[espn.team.eq("Sporting CP") & ~espn.type.eq("own-goal")].copy()
F = features_espn(espn)
bad = espn.fx.eq(0).values

out = {}
for label, mens in [("all_competitions", False), ("mens_elite_only", True)]:
    raw, pen_rate = pool(raw_all, mens)
    m, iso, stats = fit(raw)
    xg = iso.transform(m.predict_proba(F[MODEL_FEATURES])[:, 1])
    xg = np.where(F.is_penalty == 1, pen_rate, xg)
    xg = np.where(bad, np.nan, xg)
    stats.update(pool_shots=len(raw), penalty_rate=pen_rate, sporting_total_xg=float(np.nansum(xg)))
    out[label] = stats
    espn[f"xg_{label}"] = xg
    print(f"{label:18s} pool {len(raw):6d}  base {stats['base_rate']:.4f}  "
          f"test logloss {stats['log_loss']:.4f}  AUC {stats['roc_auc']:.3f}  "
          f"r_vs_SB {stats['r_vs_statsbomb']:.3f}  Sporting xG {stats['sporting_total_xg']:.2f}")

json.dump(out, open("metrics_mensonly.json", "w"), indent=2)
d = espn.xg_all_competitions - espn.xg_mens_elite_only
print(f"\nper-shot delta: mean {d.mean():+.4f}  max |delta| {d.abs().max():.3f}  "
      f"corr {espn.xg_all_competitions.corr(espn.xg_mens_elite_only):.4f}")
print("FotMob provider total for the same campaign: 16.55 (21 goals)")
