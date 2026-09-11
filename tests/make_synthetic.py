"""Vyrobí syntetické súbory vo formáte tennis-data.co.uk na otestovanie celej pipeline bez internetu.

Hráči majú skrytú "skutočnú" silu (celkovú + podľa povrchu), ktorá sa v čase mení.
Kurzy stávkových kancelárií = skutočná pravdepodobnosť + šum + marža.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

FIRST = ["Adam", "Boris", "Carlos", "Daniel", "Emil", "Filip", "Gael", "Hugo", "Ivan", "Jakub", "Karol", "Lukas",
         "Marek", "Norbert", "Oskar", "Pavol", "Radek", "Samuel", "Tomas", "Viktor", "Alex", "Felix", "Jannik", "Tomas Martin"]
LAST = ["Novak", "Kovac", "Horvath", "Varga", "Toth", "Nagy", "Balaz", "Molnar", "Szabo", "Lukac", "Klein", "Polak",
        "Mraz", "Hudak", "Kral", "Urban", "Sykora", "Blaho", "Auger-Aliassime", "de Minaur", "Bautista Agut", "Etcheverry",
        "Zverev", "Medvedev", "Rublev", "Ruud", "Fritz", "Paul", "Shelton", "Draper", "Mensik", "Lehecka", "Machac"]


def make_players(rng, tour, n=320):
    n = min(n, len(LAST) * len({f[0] for f in FIRST}))
    players, used = [], set()
    while len(players) < n:
        f, l = rng.choice(FIRST), rng.choice(LAST)
        init = "".join(t[0] + "." for t in f.split())
        if (l, init[0]) in used:  # žiadne dve rovnaké "Priezvisko I."
            continue
        used.add((l, init[0]))
        players.append({
            "full": f"{f} {l}", "td": f"{l} {init}", "skill": rng.normal(0, 1.0),
            "surf": {"Hard": rng.normal(0, 0.35), "Clay": rng.normal(0, 0.45), "Grass": rng.normal(0, 0.4)},
            "debut": rng.integers(2008, 2025),
        })
    return players


def true_p(a, b, surface, best_of):
    d = (a["skill"] + a["surf"][surface]) - (b["skill"] + b["surf"][surface])
    p = 1 / (1 + np.exp(-1.1 * d))
    if best_of == 5:
        q = p  # hrubá aproximácia: favorit v bo5 silnejší
        p = 1 / (1 + np.exp(-1.35 * d))
    return p


def generate(out_dir: str, start=2011, end=None, seed=1, bookie_noise=0.25, margin=0.05):
    end = end or dt.date.today().year
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    tourneys = [("Melbourne", "Australian Open", "Hard", 1, "Grand Slam"), ("Doha", "Qatar Open", "Hard", 2, "ATP500"),
                ("Indian Wells", "BNP Paribas Open", "Hard", 3, "Masters 1000"), ("Monte Carlo", "Monte Carlo Masters", "Clay", 4, "Masters 1000"),
                ("Madrid", "Mutua Madrid Open", "Clay", 5, "Masters 1000"), ("Paris", "French Open", "Clay", 5, "Grand Slam"),
                ("Halle", "Halle Open", "Grass", 6, "ATP500"), ("London", "Wimbledon", "Grass", 7, "Grand Slam"),
                ("Hamburg", "Hamburg Open", "Clay", 7, "ATP500"), ("Montreal", "Canadian Open", "Hard", 8, "Masters 1000"),
                ("New York", "US Open", "Hard", 8, "Grand Slam"), ("Beijing", "China Open", "Hard", 9, "ATP500"),
                ("Shanghai", "Shanghai Masters", "Hard", 10, "Masters 1000"), ("Paris", "Paris Masters", "Hard", 10, "Masters 1000")]
    for tour in ["ATP", "WTA"]:
        players = make_players(rng, tour)
        for year in range(start, end + 1):
            for p in players:
                p["skill"] += rng.normal(0, 0.15)  # sila sa v čase mení
            active = [p for p in players if p["debut"] <= year]
            ranked = sorted(active, key=lambda p: -p["skill"])
            rank = {p["full"]: i + 1 for i, p in enumerate(ranked)}
            rows = []
            for loc, name, surf, month, series in tourneys:
                if year == end and month > dt.date.today().month - 1:
                    continue
                gs = series == "Grand Slam"
                best_of = 5 if (gs and tour == "ATP") else 3
                size = 64 if gs else 32
                draw = list(rng.choice(ranked[:160], size=size, replace=False))
                day = dt.datetime(year, month, 3)
                rnd_names = ["1st Round", "2nd Round", "3rd Round", "4th Round", "Quarterfinals", "Semifinals", "The Final"][-int(np.log2(size)):]
                ri = 0
                while len(draw) > 1:
                    nxt = []
                    for i in range(0, len(draw), 2):
                        a, b = draw[i], draw[i + 1]
                        p = true_p(a, b, surf, best_of)
                        a_wins = rng.random() < p
                        w, l = (a, b) if a_wins else (b, a)
                        pw = p if a_wins else 1 - p
                        # stávková kancelária: odhad so šumom + marža
                        logit = np.log(pw / (1 - pw)) + rng.normal(0, bookie_noise)
                        bw = 1 / (1 + np.exp(-logit))
                        oddsw = 1 / (bw * (1 + margin))
                        oddsl = 1 / ((1 - bw) * (1 + margin))
                        comment = "Completed"
                        u = rng.random()
                        if u < 0.01:
                            comment = "Walkover"
                        elif u < 0.035:
                            comment = "Retired"
                        rows.append({
                            tour: 1, "Location": loc, "Tournament": name, "Date": day + dt.timedelta(days=ri),
                            ("Series" if tour == "ATP" else "Tier"): series, "Court": "Outdoor", "Surface": surf,
                            "Round": rnd_names[ri], "Best of": best_of, "Winner": w["td"], "Loser": l["td"],
                            "WRank": rank[w["full"]], "LRank": rank[l["full"]] if rng.random() > 0.02 else "NR",
                            "Wsets": 2, "Lsets": 0, "Comment": comment,
                            "B365W": round(oddsw * 0.99, 2), "B365L": round(oddsl * 0.99, 2),
                            "PSW": round(1 / (bw * 1.025), 2), "PSL": round(1 / ((1 - bw) * 1.025), 2),
                            "MaxW": round(oddsw * 1.04, 2), "MaxL": round(oddsl * 1.04, 2),
                            "AvgW": round(oddsw, 2), "AvgL": round(oddsl, 2),
                        })
                        nxt.append(w)
                    draw = nxt
                    ri += 1
            pd.DataFrame(rows).to_excel(os.path.join(out_dir, f"{tour}_{year}.xlsx"), index=False)
        pd.DataFrame(players)[["full", "td", "skill"]].to_csv(os.path.join(out_dir, f"_players_{tour}.csv"), index=False)


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
