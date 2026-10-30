import json, subprocess
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

def day(d):
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard?dates={d:%Y%m%d}"
    raw = subprocess.run(["curl","-s","--compressed","-m","30",url],capture_output=True,text=True).stdout
    try: return json.loads(raw).get("events", [])
    except Exception: return []

days = [date(2025,9,1)+timedelta(n) for n in range(0, 270)]
with ThreadPoolExecutor(8) as ex: res = list(ex.map(day, days))
seen = {}
for evs in res:
    for e in evs:
        if "Sporting" in e["name"]: seen[e["id"]] = e
for e in sorted(seen.values(), key=lambda e: e["date"]):
    c = e["competitions"][0]["competitors"]
    print(e["id"], e["date"][:10], e["name"], [x["score"] for x in c], e["status"]["type"]["description"])
print(len(seen), "Sporting matches")
