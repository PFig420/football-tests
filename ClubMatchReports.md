# ClubMatchReports.md: xG and match reports for lower-league clubs

An idea brief, not a build spec. It describes a product built from this directory's xG model and the tracking pipeline in `~/football_analysis`, what is missing, and how to check feasibility and demand before building.

Written 2026-10-28.

## 1. The idea

After each match, a club below the level that pays for Wyscout or StatsBomb gets a short report:
- **Shots:** a shot map with xG per shot, xG totals for both teams, and xG against goals over the season.
- **From the match video,** where it works: possession, distance covered and top speed per player, and team shape.
- **Season view:** xG for and against per match, per player shots and xG.

Targets: Portuguese lower leagues (Campeonato de Portugal, Liga 3, the district leagues) and youth academies.

## 2. Who might pay (hypotheses, none tested)

| Buyer | Why they might pay |
|---|---|
| Coaches and technical staff | Analysis they can't afford from the big providers, for opposition and their own team |
| Youth academies | Something to show parents, and evidence for player development |
| Players and parents | A player profile with numbers to send to scouts. Possibly a better payer than the club |

## 3. What already exists

| Part | Where | State |
|---|---|---|
| Shot-level xG model | `train_xg.py`, `xg_model.pkl` (`REPORT.md`) | HGB + isotonic, trained on 95,190 StatsBomb open-data shots. Held-out AUC 0.78, well calibrated. Against FotMob on Sporting's 12 UCL matches: per-shot r = 0.83, per-match r = 0.95 |
| Feature schema | `features.py` | 11 columns from shot location and context (header, set piece, corner, fast break, assist type) |
| Shot maps | `shot_map.py` | mplsoccer, marker size by xG |
| Detection and tracking | `~/football_analysis` | YOLO + ByteTrack; players, goalkeepers, referees and ball as separate classes |
| Team assignment | `~/football_analysis` | k-means on shirt colour, majority vote per track |
| Camera movement and pitch metres | `~/football_analysis` | Full homography; perspective transform to metres; `calibrate_pitch.py` |
| Per-player report | `main.py --report report.json` | Distances, speeds, coverage, possession |

## 4. What is missing (the hard part)

1. **Shot data.** Lower-league matches have no ESPN or FotMob events, so shots must come from somewhere else. **The simplest option is a tagging page:** someone at the club clicks each shot's location and outcome, about 20–30 shots and 15 minutes per match, filling the same 11 features as `features.py`. Automatic shot detection from video is a much bigger job.
2. **Amateur video.** The YOLO weights and the sample clip are broadcast-style. Club footage is usually a single panning camera, lower resolution and often far from the pitch. Ball detection is likely to suffer first. Every ground also needs its own pitch calibration (`README.md`, "Calibrate the pitch").
3. **Processing time.** The homography alone costs about 28 s per 750 frames of 1080p, so about 84 minutes for a 90-minute match at 25 fps, before detection, which is slower. On this laptop (7.5 GiB, CPU only) that is hours per match; a GPU or a lower frame rate is needed.
4. **The xG model's training data.** It was trained on elite professional football (with a mix of eras and levels, `REPORT.md`). Finishing and goalkeeping at district level may differ, and there is no lower-league data to check it against.
5. **Minors and consent.** Youth football means filming and profiling minors. Reports for academies need consent and GDPR handling from the start.

## 5. Competition (to check before building)

Veo, Pixellot and Hudl already sell cameras plus automatic analysis to amateur and youth clubs. Before building anything:
- Find out which stats each one includes at the club tier, and what it costs.
- Ask clubs that already have a Veo or similar camera what they still lack.

The possible gap: **analysis on footage the club already has** (camera exports, phone recordings), with no hardware to buy, and xG that is checked against a provider (§3).

## 6. Order of work

1. **Feasibility, one day:** run `~/football_analysis/main.py` on one real lower-league clip, ideally from a Veo export or a phone. Write down what works: player detection, ball, team colours, calibration.
   - **Stop rule:** if players and ball are mostly lost, drop the video half and keep the product to tagged shots and xG.
2. **Demand:** do one local club's match by hand, with tagged shots, the xG report and whatever the video gave. Give it to them for free.
3. **Ask for the next match,** and then for a price for the rest of the season.
   - **Stop rule:** if 2 of 3 clubs shown a report don't ask for another one, stop.
4. Only then build the tagging page and the report generator.

## 7. Open questions

- Club or player as the customer? The two need different reports.
- Who tags the shots: the club, a volunteer, or you as a paid service?
- Is xG enough on its own, without video, to be worth paying for at this level?
