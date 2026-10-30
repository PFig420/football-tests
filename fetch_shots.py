import json, subprocess, csv
ids = """757670 757685 757706 757710 757741 757744 757768 757781 401862583 401862584 401862888 401862890""".split()
rows = []
for i in ids:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/summary?event={i}"
    d = json.loads(subprocess.run(["curl","-s","--compressed","-m","30",url],capture_output=True,text=True).stdout)
    comps = d["header"]["competitions"][0]
    date = comps["date"][:10]
    teams = {c["team"]["displayName"]: c["homeAway"] for c in comps["competitors"]}
    title = d["header"]["gameNote"] if "gameNote" in d["header"] else ""
    for c in d.get("commentary", []):
        p = c.get("play", {})
        t = p.get("type", {}).get("type", "")
        if t.startswith("shot") or t in ("goal","penalty---scored","own-goal","penalty---saved") or "goal" in t:
            rows.append(dict(event=i, date=date, match=" vs ".join(teams), type=t, team=p.get("team",{}).get("displayName"),
                minute=p.get("clock",{}).get("displayValue"), text=p.get("text"),
                fx=p.get("fieldPositionX"), fy=p.get("fieldPositionY"), f2x=p.get("fieldPosition2X"), f2y=p.get("fieldPosition2Y"),
                gx=p.get("goalPositionX"), gy=p.get("goalPositionY")))
with open("shots.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(len(rows))
