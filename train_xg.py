"""Baseline LR vs HistGradientBoosting on shared features; isotonic calibration; grouped-by-match evaluation."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import MODEL_FEATURES, SCHEMA, features_espn, features_statsbomb, assert_same_schema

BASE_FEATURES = ["distance", "angle", "is_header", "from_corner", "from_fast_break", "from_set_piece"]
raw = pd.read_parquet("training_shots.parquet")
print("raw shots:", len(raw))
raw = raw[raw.x.notna()]; print("after dropping missing location:", len(raw))
pen = raw[raw.shot_type.eq("Penalty")]
pen_rate = float(pen.shot_outcome.eq("Goal").mean())
print(f"penalties: {len(pen)} (conversion {pen_rate:.4f}) -> excluded from training, rate saved")
raw = raw[~raw.shot_type.eq("Penalty")]; print("after dropping penalties:", len(raw))
fk = raw.shot_type.eq("Free Kick")
print(f"direct free kicks: {fk.sum()} -> excluded (ESPN text cannot identify them)")
raw = raw[~fk]; print("after dropping free kicks:", len(raw))
raw = raw[raw.x < 120.01]; raw = raw[raw.x >= 60]; print("after keeping attacking-half shots:", len(raw))
raw = raw.reset_index(drop=True)

X = features_statsbomb(raw)
sample = pd.read_csv("shots.csv")
assert_same_schema(X, features_espn(sample))
print("schema identical:", list(X.columns))
y = raw.shot_outcome.eq("Goal").astype(int).values
sb_xg = raw.shot_statsbomb_xg.astype(float).values
g = raw.match_id.values
print(f"final training pool: {len(y)} shots, {g.size and len(set(g))} matches, base rate {y.mean():.4f}")

# 60/20/20 split by match: fit / calibrate+select / test
outer = GroupShuffleSplit(1, test_size=0.2, random_state=0)
rest, te = next(outer.split(X, y, g))
inner = GroupShuffleSplit(1, test_size=0.25, random_state=1)
tr_i, ca_i = next(inner.split(X.iloc[rest], y[rest], g[rest]))
tr, ca = rest[tr_i], rest[ca_i]
assert not (set(g[tr]) & set(g[ca])) and not (set(g[rest]) & set(g[te]))
print("split sizes fit/cal/test:", len(tr), len(ca), len(te))

models = {
    "baseline_lr": (BASE_FEATURES, make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
    "lr_all": (MODEL_FEATURES, make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
    "hgb": (MODEL_FEATURES, HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                            l2_regularization=1.0, min_samples_leaf=100, random_state=0)),
}
res, fitted = {}, {}
for name, (feats, m) in models.items():
    m.fit(X.iloc[tr][feats], y[tr])
    fitted[name] = m
    p = m.predict_proba(X.iloc[ca][feats])[:, 1]
    res[name] = {"cal_split_logloss": log_loss(y[ca], p)}
    print(name, "calibration-split log loss (uncalibrated):", round(res[name]["cal_split_logloss"], 5))
best = min(res, key=lambda k: res[k]["cal_split_logloss"])
# the spec says ship the baseline unless the challenger wins; any model that loses to baseline_lr is not chosen
print("selected:", best)
feats, m = models[best]
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.99)
iso.fit(m.predict_proba(X.iloc[ca][feats])[:, 1], y[ca])

def predict(Xdf, mdl=m, f=feats, cal=iso):
    return cal.transform(mdl.predict_proba(Xdf[f])[:, 1])

yt = y[te]; base = np.full(len(te), y[tr].mean())
metrics = {"n_train": int(len(tr)), "n_cal": int(len(ca)), "n_test": int(len(te)), "selected_model": best,
           "train_base_rate": float(y[tr].mean()), "penalty_conversion_rate": pen_rate,
           "selection_cal_logloss": {k: v["cal_split_logloss"] for k, v in res.items()}, "test": {}}
def score(name, p):
    metrics["test"][name] = {"log_loss": log_loss(yt, p), "brier": brier_score_loss(yt, p), "roc_auc": roc_auc_score(yt, p) if name != "base_rate" else 0.5}
score("base_rate", base)
for name, (f, mdl) in models.items():
    score(name + "_uncalibrated", mdl.predict_proba(X.iloc[te][f])[:, 1])
p_te = predict(X.iloc[te])
score("selected_calibrated", p_te)
ok = ~np.isnan(sb_xg[te])
metrics["vs_statsbomb_xg"] = {"n": int(ok.sum()),
    "pearson_r": float(np.corrcoef(p_te[ok], sb_xg[te][ok])[0, 1]),
    "spearman_r": float(pd.Series(p_te[ok]).corr(pd.Series(sb_xg[te][ok]), method="spearman")),
    "mean_abs_diff": float(np.abs(p_te[ok] - sb_xg[te][ok]).mean()),
    "mean_ours": float(p_te[ok].mean()), "mean_sb": float(sb_xg[te][ok].mean()),
    "statsbomb_xg_logloss": float(log_loss(yt[ok], np.clip(sb_xg[te][ok], 1e-4, 1 - 1e-4))),
    "statsbomb_xg_roc_auc": float(roc_auc_score(yt[ok], sb_xg[te][ok]))}
metrics["test_total"] = {"goals": int(yt.sum()), "our_xg": float(p_te.sum())}
# calibration by decile of prediction
bins = pd.qcut(p_te, 10, duplicates="drop")
cal = pd.DataFrame({"p": p_te, "y": yt}).groupby(bins, observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
metrics["calibration_deciles"] = cal.reset_index(drop=True).to_dict("list")
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "calibration_deciles"}, indent=2))
print(cal)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot([0, cal.pred.max()], [0, cal.pred.max()], "k--", lw=1, label="perfect")
ax[0].plot(cal.pred, cal.obs, "o-", label=f"{best} + isotonic")
ax[0].set(xlabel="mean predicted xG (decile)", ylabel="observed goal rate", title="Calibration, held-out matches (10 bins)"); ax[0].legend()
ax[1].scatter(sb_xg[te][ok], p_te[ok], s=3, alpha=.25)
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set(xlabel="StatsBomb xG", ylabel="our xG", title=f"vs StatsBomb xG (r={metrics['vs_statsbomb_xg']['pearson_r']:.2f})", xlim=(0, 1), ylim=(0, 1))
fig.tight_layout(); fig.savefig("calibration.png", dpi=130)

pickle.dump({"model": m, "iso": iso, "features": feats, "penalty_rate": pen_rate, "name": best}, open("xg_model.pkl", "wb"))
print("saved xg_model.pkl, metrics.json, calibration.png")
