# Sporting CP 2025/26 UCL — shot-level xG model

Fallback model: no provider xG is reachable for these matches (CLAUDE.md §3). If a StatsBomb/Opta key or FBref access appears, prefer the real xG.

## What was built
| File | Purpose |
| --- | --- |
| `build_training_set.py` | All 3,961 free StatsBomb matches (every competition, incl. women's/youth/old finals — user's choice) → `training_shots.parquet` |
| `features.py` | `features_statsbomb()` / `features_espn()`, identical 11-column schema, `assert_same_schema` run in `train_xg.py` |
| `train_xg.py` | LR baseline, LR (all features), HistGradientBoosting; isotonic calibration; grouped-by-match evaluation → `xg_model.pkl`, `metrics.json`, `calibration.png` |
| `apply_xg.py` | `shots.csv` → `shots_xg.csv`, `match_xg_totals.csv` (Sporting's shots only — user's choice) |
| `shot_map.py` | `X_SCALE` fix was already present; marker area now ∝ xG |

## Training data and filters (every count printed by `train_xg.py`)
100,864 shots → 0 missing location → −1,118 penalties (conversion **76.3 %**, used for ESPN penalties) → −4,449 direct free kicks (ESPN text never says "free kick", 0/309, so they cannot be flagged) → −107 shots outside the attacking half → **95,190 shots, 3,961 matches, base rate 10.4 %**.
Mixing all eras/levels/genders shifts the base rate and shot mix versus a modern men's UCL; I did not test a restricted set.
Split by match: 57,104 fit / 19,131 calibration+selection / 18,955 test.

## Held-out results (test split, 18,955 shots, 1,984 goals)
| Model | Log loss | Brier | ROC AUC |
| --- | --- | --- | --- |
| Always base rate | 0.3352 | 0.0937 | 0.500 |
| LR baseline (distance, angle, header, set-piece flags) | 0.2876 | 0.0824 | 0.770 |
| LR all features | 0.2844 | 0.0816 | 0.777 |
| HGB (selected on calibration split) | 0.2827 | 0.0814 | 0.781 |
| **HGB + isotonic (shipped)** | **0.2832** | 0.0815 | 0.781 |

- Beats base-rate log loss by 0.052. HGB beats the baseline by only 0.004 — a small gain, as expected.
- Isotonic calibration made held-out log loss *slightly worse* (0.2827 → 0.2832); the uncalibrated HGB was already well calibrated. I kept it for the spec'd pipeline; the difference is negligible.
- AUC 0.78 is inside the expected 0.75–0.80 band; no sign of a leak.
- **Calibration** (`calibration.png`): the 10 decile points sit on the diagonal (e.g. 0.209 predicted vs 0.209 observed; top bin 0.409 vs 0.398). Test total xG 1,999.9 vs 1,984 goals. Isotonic output is step-valued, hence the horizontal bands in the right-hand scatter.
- **Agreement with StatsBomb xG** (same test shots): Pearson r = **0.81**, Spearman 0.88, mean abs diff 0.041, mean 0.106 vs 0.101. StatsBomb's own xG (richer features) scores log loss 0.2677 / AUC 0.810, so that is the ceiling our location-plus-text features don't reach.

## Applying to ESPN (`shots_xg.csv`)
- 309 rows → 1 own goal removed → 156 opponent shots not modelled → 152 Sporting rows; **3 have `fx == 0` → `xg = NaN`**; 149 have xG. Sporting's 3 penalties (2 scored, 1 saved) are assigned 0.763 each; the 4th penalty in `shots.csv` is Bodø/Glimt's and was not modelled. (Per-match shot counts in `match_xg_totals.csv` include the NaN rows; the map shows only the 149.)
- ESPN regex counts verified against text: header 41, "following a corner" 31, "following a fast break" 16, "following a set piece" 9, "Assisted by" 247, "with a cross" 47, "with a through ball" 7, "free kick" 0.
- Mapping caveats: ESPN "following a set piece" ≈ StatsBomb `From Free Kick` (assumed); ESPN states "following a corner" only on assisted shots; unassisted shots with no body-part phrase ("with an attempt from a difficult angle") are treated as non-headers.

### Per-match xG vs goals (Sporting)
| Date | Match | xG | Goals |
| --- | --- | --- | --- |
| 2025-09-18 | Sporting–Kairat | 2.95 | 4 |
| 2025-10-01 | Napoli–Sporting | 1.34 | 1 |
| 2025-10-22 | Sporting–Marseille | 1.07 | 2 |
| 2025-11-04 | Juventus–Sporting | 0.24 | 1 |
| 2025-11-26 | Sporting–Club Brugge | 2.19 | 3 |
| 2025-12-09 | Bayern–Sporting | 0.49 | 0 (+1 own goal, excluded) |
| 2026-01-20 | Sporting–PSG | 0.89 | 2 |
| 2026-01-28 | Athletic–Sporting | 1.13 | 3 |
| 2026-03-11 | Bodø/Glimt–Sporting | 0.76 | 0 |
| 2026-03-17 | Sporting–Bodø/Glimt (**extra time**) | 4.14 | 5 |
| 2026-04-07 | Sporting–Arsenal | 1.20 | 0 |
| 2026-04-15 | Arsenal–Sporting | 0.34 | 0 |
| **Total** | | **16.74** | **21** |

Total xG is 80 % of the 21 goals — inside the "not half or double" band, but Sporting finished above the model (≈+4.3 goals). The 3 penalties are assigned 2.29 xG for 2 goals, so the gap is all open play: 14.45 xG for 19 goals. With 149 shots this is within plausible variance, but also consistent with the model being slightly low for this team (cross-source error, or Sporting's finishing). Match-level xG from ~10 shots has a very wide error bar; do not read the matches individually. The Bodø/Glimt second leg includes extra time and should not be treated as a 90-minute rate.

### Sanity checks
- No open-play shot above 0.6 xG. Highest goal xG 0.76 (penalties); lowest goal xG 0.024.
- The only six-yard-box/very-close shot under 0.1 is a Matheus Reis header against Bayern (xG 0.05, 9.5 m from goal centre) — a header, so low is reasonable.

## Validation against provider xG (added after the build)

FotMob's match pages turn out to embed a full shot map with per-shot xG in their `__NEXT_DATA__`
blob, reachable with a plain `curl` (`fetch_fotmob.py`, `fotmob_extract.py`). All 12 matches were
pulled: **315 shots, 155 of them Sporting's**, with provider xG, coordinates, `situation` and
`shotType`. `compare_xg.py` matches them to our shots on date + minute (±1) + player surname and
writes `xg_comparison.csv` and `validation_vs_fotmob.png`.

This measures the cross-source error that was previously listed as unmeasured.

| | Ours | FotMob |
| --- | --- | --- |
| Sporting total xG, 12 matches | 16.74 | **16.55** |
| Per-match correlation (n=12) | | **r = 0.95** |
| Per-shot correlation (n=146 matched of 149) | | **r = 0.83**, Spearman 0.87 |
| Per-shot mean abs diff / bias / RMSE | | 0.043 / **+0.008** / 0.083 |
| Penalty value | 0.763 | 0.788 |

- Agreement with a real provider on these exact shots (r = 0.83) is **as good as the model's
  agreement with StatsBomb on the held-out StatsBomb split** (r = 0.81). The ESPN→StatsBomb
  coordinate/text mapping is not degrading the model as much as feared.
- **Sporting's overperformance is real, not a model artefact:** FotMob independently has them at
  16.55 xG for 21 goals. At σ = 3.53 that is z = 1.21 (~14 % likely), i.e. unremarkable variance.
- **Where we lose:** the model cannot see big chances. Our top end compresses — the four largest
  disagreements are all shots FotMob rates 0.55–0.76 and we rate 0.21–0.30 (e.g. Trincão, Kairat,
  80'). That is exactly the freeze-frame information (keeper position, defenders, one-on-one) that
  ESPN commentary does not carry. We run slightly high on corners (0.095 vs 0.068, n=22).
- FotMob also gives **xG against** (opponents: 160 shots, 13.48 xG), which our model never produced.

**Recommendation: use FotMob's xG for these matches and keep this model as the fallback** for
fixtures FotMob does not cover. The model is validated, but it is strictly less informed.

### Training-mix sensitivity (the concern that did not pan out)

`sensitivity_training_mix.py` retrains on men's elite club + international shots only (59,120 of
95,190 — dropping all women's competitions, Indian Super League, MLS, U20) because the full pool
mixes competitions whose conversion rates differ (Frauen Bundesliga 0.123 vs Serie A 0.095).

| Pool | Shots | Base rate | Test log loss | AUC | r vs StatsBomb | Sporting total xG |
| --- | --- | --- | --- | --- | --- | --- |
| All competitions (shipped) | 95,190 | 0.1039 | 0.2832 | 0.781 | 0.807 | **16.74** |
| Men's elite only | 59,120 | 0.1024 | 0.2851 | 0.792 | 0.821 | 17.22 |

Per-shot correlation between the two is 0.990, max difference 0.091. The restricted model moves
Sporting's total *away* from FotMob's 16.55, not toward it. **The training mix does not matter here
— keep the shipped model.** Base rates barely differ once you condition on location and situation.

## Assumptions and limitations
1. **y-scale (§4) is not resolved.** I assumed 68 m per `fy` unit. Six "difficult angle" shots sit at lateral offsets of 10–13 m (68) or 12–15 m (80), 2–5 m from the goal line: both plausible, so they cannot discriminate. Sensitivity of using 80 m: total Sporting xG 16.74 → 15.24, per-shot correlation 0.98, mean |Δ| 0.014, max |Δ| 0.19.
2. ~~Cross-source error was not measured.~~ **Now measured** against FotMob: per-shot r = 0.83, bias +0.008, MAE 0.043. See the validation section.
3. Direct free kicks are excluded from training and cannot be identified in ESPN, so any ESPN direct free kick is valued like an open-play shot from that spot (likely too high).
4. Goal-mouth placement (`gx/gy`) is unpopulated; `f2x/f2y` were not used.
5. Opponents' 156 shots were not modelled (user's choice), so there is no xG-against.
6. 149 shots is one campaign: do not rank players on this.
