"""Match our ESPN-based xG to FotMob's provider xG shot by shot (minute + player surname)."""
import re, unicodedata
import pandas as pd, numpy as np

def strip(s):  # accent-insensitive compare: "Trincão" in ESPN vs FotMob
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()

ours = pd.read_csv("shots_xg.csv")
ours = ours[ours.xg.notna()].copy()
ours["min"] = ours.minute.astype(str).str.extract(r"(\d+)").astype(float)
ours["date"] = ours.date.astype(str)

fm = pd.read_csv("fotmob_shots.csv")
fm = fm[fm.team == "Sporting CP"].copy()
fm["date"] = fm.matchDate.astype(str)

rows, used = [], set()
for _, o in ours.iterrows():
    cand = fm[(fm.date == o.date) & (abs(fm["min"] - o["min"]) <= 1) & (~fm.id.isin(used))]
    hit = None
    for _, c in cand.iterrows():                       # prefer a surname match
        if strip(c.playerName).split()[-1] in strip(o.text):
            hit = c; break
    if hit is None and len(cand) == 1:                 # else accept a lone minute match
        hit = cand.iloc[0]
    if hit is not None:
        used.add(hit.id)
        rows.append(dict(date=o.date, min=o["min"], player=hit.playerName, ours=o.xg,
                         fotmob=hit.expectedGoals, situation=hit.situation,
                         shotType=hit.shotType, event=hit.eventType, text=o.text[:70]))

m = pd.DataFrame(rows)
m.to_csv("xg_comparison.csv", index=False)
d = m.ours - m.fotmob
print(f"matched {len(m)} of {len(ours)} our shots to {len(fm)} FotMob shots")
print(f"ours total {m.ours.sum():.2f} | fotmob total {m.fotmob.sum():.2f}")
print(f"Pearson r {m.ours.corr(m.fotmob):.3f} | Spearman {m.ours.corr(m.fotmob, method='spearman'):.3f}")
print(f"mean abs diff {d.abs().mean():.3f} | bias (ours-fotmob) {d.mean():+.3f} | RMSE {np.sqrt((d**2).mean()):.3f}")
print("\nworst 8 disagreements:")
print(m.assign(diff=d).reindex(d.abs().sort_values(ascending=False).index)
       .head(8)[["date","min","player","ours","fotmob","diff","situation","shotType"]].to_string(index=False))
print("\nby situation:")
print(m.groupby("situation").agg(n=("ours","size"), ours=("ours","mean"), fotmob=("fotmob","mean")).round(3).to_string())

# --- validation plot -------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].scatter(m.fotmob, m.ours, s=22, alpha=.6, edgecolors="k", linewidths=.3)
ax[0].plot([0, .8], [0, .8], "k--", lw=1)
ax[0].set(xlabel="FotMob xG (provider)", ylabel="our xG", xlim=(0, .85), ylim=(0, .85),
          title=f"Per shot, n={len(m)}  r={m.ours.corr(m.fotmob):.2f}  MAE={abs(m.ours-m.fotmob).mean():.3f}")
t = m.groupby("date").agg(ours=("ours", "sum"), fotmob=("fotmob", "sum"))
ax[1].scatter(t.fotmob, t.ours, s=60, edgecolors="k", linewidths=.4)
lim = max(t.max()) * 1.1
ax[1].plot([0, lim], [0, lim], "k--", lw=1)
ax[1].set(xlabel="FotMob xG (provider)", ylabel="our xG", xlim=(0, lim), ylim=(0, lim),
          title=f"Per match, n={len(t)}  r={t.ours.corr(t.fotmob):.2f}")
fig.suptitle("Our ESPN-based xG vs FotMob provider xG — Sporting CP, 2025/26 UCL")
fig.tight_layout(); fig.savefig("validation_vs_fotmob.png", dpi=130)
print("saved validation_vs_fotmob.png")
