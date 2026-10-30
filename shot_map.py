import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

d = pd.read_csv("shots_xg.csv")  # from apply_xg.py: Sporting shots only, own goal removed, has xg
d = d[(d.fx > 0) & d.xg.notna()]
# ESPN coords: fx = distance from goal line in units of 48.1 m (calibrated from penalties and from
# the in-box/outside-box split in the commentary text; see CLAUDE.md §4). fy = lateral, 0.5 = centre,
# < 0.5 = attacking left. The 68 m width scale is an assumption, not calibrated.
X_SCALE, Y_SCALE = 48.1, 68
d["x"] = 105 - X_SCALE * d.fx
d["y"] = 34 + (d.fy - 0.5) * Y_SCALE
style = {  # outcome -> (label, color, marker, size)
    "goal": ("Goal", "#1b9e3f", "*", 260), "shot-on-target": ("On target", "#1f77b4", "o", 70),
    "shot-off-target": ("Off target", "#999999", "o", 60), "shot-blocked": ("Blocked", "#d62728", "x", 60),
    "shot-hit-woodwork": ("Woodwork", "#ff7f0e", "D", 70)}
d["size"] = 25 + 700 * d.xg  # marker area ~ xG
d["kind"] = d.type.replace({"goal---header": "goal", "penalty---scored": "goal", "penalty---saved": "shot-on-target"})

matches = d.drop_duplicates("event").sort_values("date")[["event", "date", "match"]]
pitch = VerticalPitch(half=True, pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#555", pad_bottom=0.5)
fig, axes = plt.subplots(3, 4, figsize=(18, 14))
for ax, (_, m) in zip(axes.flat, matches.iterrows()):
    pitch.draw(ax=ax)
    s = d[d.event == m.event]
    for k, (lab, c, mk, sz) in style.items():
        q = s[s.kind == k]
        if len(q): pitch.scatter(q.x, q.y, ax=ax, c=c, marker=mk, s=q["size"] * (1.6 if mk == "*" else 1), edgecolors="k" if mk != "x" else None, linewidths=.6, zorder=3)
    g = (s.kind == "goal").sum()
    ax.set_title(f"{m.date}\n{m.match.replace(' vs ', ' / ')}\n{len(s)} shots, {g} goals, xG {s.xg.sum():.2f}", fontsize=10)
handles = [plt.Line2D([], [], marker=mk, color=c, linestyle="", markersize=9, label=lab) for lab, c, mk, _ in style.values()]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12)
fig.suptitle("Sporting CP shots - 2025/26 Champions League (attacking up; marker size ~ xG; ESPN data)", fontsize=16)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig("sporting_cl_shotmaps.png", dpi=130)
print(len(d), "shots plotted")
