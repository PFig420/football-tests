# xG model for Sporting CP — build spec

Written 2026-10-17. This file is the brief for the agent that builds the model. Everything under
"Verified facts" was checked by running code in this directory; everything under "Plan" is a
proposal you may change if you find something better, as long as you say so.

## 1. Mission

Produce a per-shot expected-goals (xG) value for every shot in `shots.csv` (Sporting CP's 2025/26
Champions League campaign, 12 matches, 309 shots by both teams, 149 of them Sporting's), because no
provider gives us xG for these matches (see §3).

The deliverable is a model plus the shot-level xG column, not a research project. Aim for a
defensible, well-calibrated baseline, not a state-of-the-art model.

## 2. Environment

- Working directory: `/home/pedro/football-tests`. Keep all code here.
- Python: `.venv/bin/python` (3.14). Installed: `soccerdata` 1.9.1, `mplsoccer`, `statsbombpy`,
  `kloppy`, `pandas`, `scikit-learn`, `matplotlib`, `numpy`.
- `uv` is vendored at `./uvbin/uv` — the system has no `pip` and no working `python3 -m venv`.
  Install anything else with `./uvbin/uv pip install --python .venv/bin/python <pkg>`.
- Always `export SOCCERDATA_DIR=$PWD/sd` before using soccerdata, or it writes to `/home/pedro/soccerdata`.
- No browser/chromedriver is installed, so any soccerdata reader needing Selenium (WhoScored) is out.
- Network: GitHub (StatsBomb open data) works. See §3 for what does not.

### Existing files

| File | What it is |
| --- | --- |
| `query_espn.py` | Finds Sporting's UCL matches by looping ESPN's scoreboard API over single dates. |
| `fetch_shots.py` | Pulls shot events from ESPN's match summary endpoint → `shots.csv`. |
| `shots.csv` | 309 shots, both teams, 12 matches. Columns in §4. |
| `shot_map.py` | mplsoccer 12-panel shot map → `sporting_cl_shotmaps.png`. |

## 3. Verified facts: who publishes per-shot xG

Checked on 2026-10-17 from this machine. The question "does per-shot xG exist from major providers"
has two separate answers: it exists almost everywhere, but **not reachable for these 12 matches**.

| Provider | Per-shot xG published? | UCL 2025/26 reachable here? | Evidence |
| --- | --- | --- | --- |
| StatsBomb open data | Yes — `shot_statsbomb_xg` on every shot | **No.** Open UCL data is finals only, 1 match per season, latest 2018/19 | `sb.competitions()` lists 18 UCL seasons; `sb.matches(16, 4)` returns exactly 1 match (the 2019 final) |
| Understat | Yes — `read_shot_events()` | No — top-5 domestic leagues only, no UCL at all | `sd.Understat.available_leagues()` |
| FBref (Opta xG) | Yes — per-shot xG in match shot logs, and FBref does cover UCL | No — `fbref.com` returns **HTTP 403** from this host; UCL is also absent from soccerdata's default league list (addable via `league_dict.json`) | direct curl with browser UA |
| Sofascore | Yes — xG in its shotmap endpoint | No — **HTTP 403** on both `api.sofascore.com` and `www.sofascore.com/api`; soccerdata's reader only exposes schedule/table anyway | direct curl; `dir(sd.Sofascore)` |
| FotMob | Yes — xG per shot | **YES — superseded, see below** | `fetch_fotmob.py` |
| WhoScored (Opta) | Partly | No — needs Selenium, no browser installed | `dir(sd.WhoScored)` |
| ESPN (our source) | **No xG of any kind** | n/a — but it is the only source that gave us these matches | `summary` endpoint has no xG field |

> **Superseded 2026-10-17.** FotMob's *match pages* (not its `/api/` endpoints, which are
> token-gated — that is what the 403-ish result above actually tested) embed the full shot map with
> per-shot xG in their `__NEXT_DATA__` blob, and a plain `curl` gets it. All 12 matches are now
> pulled to `fotmob_shots.csv` by `fetch_fotmob.py`. **Prefer that provider xG; this model is the
> fallback.** See REPORT.md's validation section — the model agrees with it at r = 0.83 per shot.
> Note `pageUrl`'s slug is shared by both legs of a two-legged tie; use `/match/<id>`.

**No major provider open-sources its production xG model.** Opta, StatsBomb, Wyscout, Understat and
FotMob all keep the model proprietary; some publish methodology prose and some publish the output
values. So we cannot borrow a model — but StatsBomb's open data gives us ~tens of thousands of
labelled shots *with their xG as a reference column*, which is the next best thing.

If the user later obtains a StatsBomb/Opta API key, or unblocks FBref via a proxy, **prefer the real
provider xG over this model** and treat this model as a fallback.

## 4. Verified facts: the ESPN data and its coordinates

`shots.csv` columns: `event, date, match, type, team, minute, text, fx, fy, f2x, f2y, gx, gy`.

- `type` ∈ {`shot-off-target` 113, `shot-blocked` 80, `shot-on-target` 69, `goal` 29,
  `shot-hit-woodwork` 9, `goal---header` 4, `penalty---scored` 3, `penalty---saved` 1, `own-goal` 1}.
- `gx`/`gy` are **always 0.0** — the goal-mouth placement fields are unpopulated. Ignore them.
- `f2x`/`f2y` are populated for roughly half of shots and look like the assist origin. Unverified.
- A few rows have `fx == 0.0`, which is missing data, not a shot from the goal line. Drop them.
- `text` is rich Opta-style commentary and is the main feature source: body part, zone, assist type,
  set-piece origin, fast break are all stated in prose.

### Coordinate calibration — do not skip this

`fx` is **distance from the goal line as a fraction of 48.1 m**, not a fraction of pitch length.
Two independent checks agree:

1. All 4 penalties sit at exactly `fx=0.230, fy=0.500` → 11.0 m / 0.230 = **47.8 m per unit**.
2. Fitting the `fx` threshold that best separates "…the box" from "outside the box" in `text` gives
   `fx=0.343` at **99.6 %** accuracy (n=271). The box edge is 16.5 m → **48.1 m per unit**, which
   back-predicts the penalty spot at 11.06 m.

So: **`distance_from_goal_m = 48.1 * fx`**.

```python
X_SCALE = 48.1              # metres per fx unit, calibrated above
dist_m = X_SCALE * df.fx    # straight-line-ish distance from the goal line
```

`fy` is lateral position, `0.5` = centre. **`fy < 0.5` is the attacking left, `fy > 0.5` the
attacking right** — verified: "left side of the box" averages `fy=0.318` (n=36), "right side of the
box" averages `fy=0.682` (n=31). This resolves the mirror ambiguity noted in the old shot map.

**Open question — the lateral scale is not pinned down.** In-box shots reach `|fy-0.5| = 0.252`.
If that extreme is the corner of the box (20.15 m from centre) the y-scale is ~80 m per unit; if the
pitch width 68 m is the right normaliser the same shot sits 17.1 m from centre, inside the box
corner, which is also plausible. Any value in ~68–84 fits. Resolve it if you can (look for shots
described as "from a difficult angle" or near the byline, n=10 mention "angle"), otherwise assume
68 m, state the assumption in the output, and check how much the angle feature moves if you use 80.

> **Known defect:** `shot_map.py` as originally written used `x = 105 * (1 - fx)`, which treats `fx`
> as a fraction of a 105 m pitch and puts every shot ~2.2× too far from goal (penalties land 24 m
> out). If the map has not already been regenerated with `X_SCALE`, fix it.

## 5. The central design constraint

The labels and the features come from different worlds:

- **Training data** (StatsBomb open) has freeze frames, body part, technique, under-pressure flags,
  one-on-one flags — ~33 shots per match, 80 free competitions, so tens of thousands of shots.
- **Application data** (ESPN) has location plus a sentence.

A model trained on StatsBomb's rich features cannot be applied to ESPN shots. **Train only on
features that both sources can produce.** Build the feature extractor as two functions that emit the
same schema:

```python
def features_statsbomb(events: pd.DataFrame) -> pd.DataFrame: ...
def features_espn(shots: pd.DataFrame) -> pd.DataFrame: ...
```

and assert the two frames have identical columns and dtypes before training.

### Proposed shared feature set

| Feature | From StatsBomb | From ESPN |
| --- | --- | --- |
| `distance` (m to goal centre) | `location` on 120×80 pitch | `48.1 * fx`, `(fy-0.5)*Y_SCALE` |
| `angle` (rad subtended by the 7.32 m goal) | same | same |
| `is_header` | `shot_body_part == 'Head'` | `"header"` in text, or `type == 'goal---header'` |
| `is_weak_foot_side` / `body_part` | `shot_body_part` | "left footed" / "right footed" in text |
| `is_penalty` | `shot_type == 'Penalty'` | `type.startswith('penalty')` |
| `is_free_kick` | `shot_type == 'Free Kick'` | "free kick" in text |
| `from_corner` | `play_pattern == 'From Corner'` | "following a corner" in text |
| `from_fast_break` | `play_pattern == 'From Counter'` | "following a fast break" in text |
| `assisted` | `shot_key_pass_id` present | "Assisted by" in text |
| `assist_was_cross` | key pass `pass_cross` | "with a cross" in text |
| `assist_was_through_ball` | key pass `pass_through_ball` | "with a through ball" in text |

Use absolute lateral offset `|y - centre|` rather than signed `y`, so the model is symmetric and a
mirror error in either source cannot hurt it.

Verify each ESPN regex against the actual strings before trusting it — print the match counts and
eyeball a sample. Do not assume a phrase exists because Opta commentary usually contains it.

## 6. Plan

1. **Build the training set.** Pull StatsBomb open events for a set of competitions. Prefer recent
   men's club and international football over 1970s finals and over youth/women's competitions if
   you want the closest match to UCL 2025/26 — but say what you chose and why, and note that mixing
   eras/levels shifts the base rate. Target ≥ 15 000 shots. Cache to parquet; the GitHub pull is slow.
2. **Label.** `y = (shot_outcome == 'Goal')`. Keep blocked shots — StatsBomb assigns them xG, and
   ESPN gives us blocked-shot origins too, so dropping them would bias the ESPN application. Drop
   own goals. Handle penalties separately (see §7).
3. **Baseline.** Logistic regression on `distance`, `angle`, `is_header`, and the set-piece flags.
   This is the Soccermatics-style baseline and it is the thing to beat.
4. **Challenger.** Gradient boosting (`HistGradientBoostingClassifier` or LightGBM if you install it).
   Expect a small gain. If it does not beat the baseline on held-out log loss, ship the baseline.
5. **Calibrate.** Fit isotonic or Platt calibration on a held-out split
   (`CalibratedClassifierCV`). xG is a probability; calibration matters more than discrimination.
6. **Evaluate**, grouped by match so no match spans the train/test split:
   - log loss and Brier score against the baseline of "always predict the base rate";
   - ROC AUC (expect ~0.75–0.80 for a location-only model; much higher means a leak);
   - a calibration curve (10 bins) saved as a PNG;
   - **agreement with StatsBomb's own xG** on the test shots — correlation and mean absolute
     difference. This is the key credibility check and the reason to use StatsBomb for training.
7. **Apply to `shots.csv`.** Emit `shots_xg.csv` with an added `xg` column, plus per-match and
   per-team xG totals for Sporting's 12 matches.
8. **Sanity-check the output.** Sporting scored 22 goals in these 12 matches: 21 from their own
   shots (`type` in `goal`, `goal---header`, `penalty---scored`) plus 1 own goal in their favour,
   which is excluded from modelling. Total xG should land in a believable band around 21, not
   at half or double it. Any single open-play shot above ~0.6 xG, or a tap-in below ~0.1, deserves a
   look at its `text`. Report the biggest discrepancies rather than hiding them.
9. **Update the shot map** so marker size scales with xG, once the `X_SCALE` fix from §4 is in.

## 7. Pitfalls

- **Penalties.** Do not let the model learn them from location — there are only 4 in the ESPN set and
  they all share one canned coordinate. Assign the historical conversion rate (~0.76–0.79, compute
  it from the StatsBomb data rather than hard-coding my number) and exclude them from training.
- **The `0.23/0.5` canned penalty coordinate** is also why §4's calibration uses the box-edge fit as
  the primary evidence: identical values across all 4 penalties suggest a constant ESPN substitutes,
  not a measurement. The two methods agreeing to within 0.3 % is what makes it trustworthy.
- **Extra time.** The 2026-03-17 Bodø/Glimt match went to extra time (36 Sporting shots, by far the
  most). Do not treat it as a 90-minute match in any per-match rate.
- **One own goal** is in `shots.csv` and is excluded from the map. This is confirmed to be the
  Sporting goal at Bayern on 2025-12-09 (Kimmich OG, Bayern 3–1 Sporting), which is why that panel
  shows 0 goals. Note ESPN tags the own-goal row with `team = Sporting CP`, i.e. the *benefiting*
  team, so a naive `groupby(team)` will miscount it as a Sporting shot. Filter it explicitly.
- **Sample size.** 149 Sporting shots is one campaign. Per-match xG from ~12 shots has a huge error
  bar. Do not present match-level xG as precise, and do not rank players on it.
- **Cross-source error.** Even a good model carries the ESPN→StatsBomb coordinate mismatch into
  every prediction. Say so wherever the numbers are presented.

## 8. Do not

- Do not scrape FBref, Sofascore or FotMob around the 403s with proxies, rotating user agents or
  scraping services. They blocked us; ask the user before trying another route.
- Do not invent xG values for shots the model cannot handle. Emit `NaN` and say how many.
- Do not quietly drop shots. Every filter gets a printed count.
- Do not report a metric you did not compute. If a step was skipped, say it was skipped.

## 9. Deliverables

```
build_training_set.py   # StatsBomb open data -> training_shots.parquet
features.py             # features_statsbomb() + features_espn(), shared schema, asserted equal
train_xg.py             # baseline + challenger, calibration, metrics -> xg_model.pkl, metrics.json
apply_xg.py             # shots.csv -> shots_xg.csv, per-match totals
shot_map.py             # updated: X_SCALE fix, marker size ~ xg
REPORT.md               # what was built, metrics, calibration plot, agreement with StatsBomb xG,
                        # the assumptions from §4, and the honest limitations
```

## 10. Acceptance criteria

- `features_espn` and `features_statsbomb` provably emit the same schema.
- Calibrated model beats the base-rate baseline on held-out log loss, with the number shown.
- Calibration curve plotted and discussed, not just saved.
- Correlation with StatsBomb's own xG reported on held-out shots.
- `shots_xg.csv` exists, every non-penalty shot has an `xg`, and the per-match totals are compared
  against actual goals in `REPORT.md`.
- The y-scale assumption from §4 is stated explicitly with its sensitivity.

## 11. Ask the user before deciding

- Which StatsBomb competitions to train on (era/level trade-off, §6.1).
- Whether a paid StatsBomb/Opta key is available — it would make this whole model unnecessary.
- Whether opponents' shots (160 of the 309) should be modelled too, or only Sporting's 149.
