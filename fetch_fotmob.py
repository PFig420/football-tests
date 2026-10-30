"""Download the 12 Sporting UCL 2025/26 match pages from FotMob and extract per-shot xG."""
import json, re, subprocess, time, sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir
from fotmob_extract import shots_from_html

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
RAW = Path("raw"); RAW.mkdir(exist_ok=True)

league = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                              open("raw/ucl2526.html", encoding="utf-8").read(), re.S).group(1))
matches = pd.DataFrame(league["props"]["pageProps"]["fixtures"]["allMatches"])
sp = matches[matches.home.apply(lambda t: "Sporting" in t["name"])
             | matches.away.apply(lambda t: "Sporting" in t["name"])]

frames, failed = [], []
for _, m in sp.iterrows():
    dest = RAW / f"fotmob_{m.id}.html"
    if not dest.exists():
        # NB: pageUrl's slug is shared by both legs of a two-legged tie and resolves to the
        # second leg only. /match/<id> is the unambiguous form.
        url = f"https://www.fotmob.com/match/{m.id}"
        r = subprocess.run(["curl", "-s", "--compressed", "-m", "60", "-A", UA, "-o", str(dest),
                            "-w", "%{http_code}", url], capture_output=True, text=True)
        print(f"{m.id} {r.stdout} {url}")
        time.sleep(2)                      # be polite: 1 request / 2s
    try:
        frames.append(shots_from_html(str(dest)))
    except Exception as e:
        failed.append((m.id, repr(e)))

shots = pd.concat(frames, ignore_index=True)
shots.to_csv("fotmob_shots.csv", index=False)
print(f"\n{len(shots)} shots from {shots.matchName.nunique()} matches -> fotmob_shots.csv")
if failed:
    print("FAILED:", failed)
