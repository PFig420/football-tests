"""Extract per-shot xG from a saved FotMob match page (__NEXT_DATA__)."""
import json, re, sys, pandas as pd

def shots_from_html(path):
    h = open(path, encoding="utf-8").read()
    d = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))
    c = d["props"]["pageProps"]["content"]
    gen = d["props"]["pageProps"].get("general", {})
    shots = pd.DataFrame(c["shotmap"]["shots"])
    names = {t["id"]: t["name"] for t in [gen.get("homeTeam", {}), gen.get("awayTeam", {})] if t}
    shots["team"] = shots.teamId.map(names)
    shots["matchName"] = f"{gen.get('homeTeam',{}).get('name')} vs {gen.get('awayTeam',{}).get('name')}"
    shots["matchDate"] = (gen.get("matchTimeUTCDate") or "")[:10]
    return shots

if __name__ == "__main__":
    s = shots_from_html(sys.argv[1])
    print(s.matchName.iloc[0], s.matchDate.iloc[0], "|", len(s), "shots")
    print(s.groupby("team").agg(shots=("expectedGoals","size"), xG=("expectedGoals","sum"),
                                goals=("eventType", lambda x: (x=="Goal").sum())).round(2).to_string())
    print()
    print(s[["min","team","playerName","situation","shotType","eventType","expectedGoals","x","y"]]
          .sort_values("min").head(12).to_string(index=False))
