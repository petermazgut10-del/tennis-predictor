"""Predikcie nadchádzajúcich zápasov z The Odds API + hľadanie value."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import config
from src import blend, markov
from src.elo import EloBook
from src.model import Calibrated, features
from src.serve import ServeBook
from src.names import FullNameIndex
from src.surfaces import best_of_for, surface_for, tour_for

MIN_MATCHES_FOR_VALUE = 15   # hráč s menej zápasmi v histórii nedostane value tip (málo dát)


def player_index(book: EloBook, aliases_path: str) -> FullNameIndex:
    players = {key: {"name": pl.name, "last": pl.last, "n": pl.n} for key, pl in book.p.items()}
    return FullNameIndex(players, aliases_path)


def snapshot(book: EloBook, ka: str, kb: str, surface: str, when: pd.Timestamp) -> dict:
    a, b = book.p[ka], book.p[kb]
    s = book.snapshot(a, b, surface, when)
    s["a_rank"], s["b_rank"] = a.rank, b.rank
    return s


def parse_odds(event: dict, home: str, away: str) -> dict:
    prices = {home: [], away: []}
    pinnacle = {}
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            outs = {o["name"]: o.get("price") for o in m.get("outcomes", [])}
            if home in outs and away in outs and outs[home] and outs[away]:
                prices[home].append((float(outs[home]), bk.get("title", bk.get("key"))))
                prices[away].append((float(outs[away]), bk.get("title", bk.get("key"))))
                if bk.get("key") == "pinnacle":
                    pinnacle = {home: float(outs[home]), away: float(outs[away])}
    out = {}
    for side in (home, away):
        lst = prices[side]
        if not lst:
            return {}
        best = max(lst, key=lambda x: x[0])
        out[side] = {
            "best": best[0], "best_book": best[1],
            "avg": float(np.mean([x[0] for x in lst])),
            "pinnacle": pinnacle.get(side),
            "n_books": len(lst),
        }
    # férová pravdepodobnosť trhu: Pinnacle bez marže; inak medián kancelárií (medián
    # zámerne ignoruje najlepší kurz, na ktorý sa stavia - inak by sa edge meral sám proti sebe)
    if pinnacle:
        i1, i2 = 1 / pinnacle[home], 1 / pinnacle[away]
        out["fair_home"] = i1 / (i1 + i2)
        out["fair_src"] = "Pinnacle"
    elif len(prices[home]) >= 3:
        m1 = float(np.median([x[0] for x in prices[home]]))
        m2 = float(np.median([x[0] for x in prices[away]]))
        out["fair_home"] = (1 / m1) / (1 / m1 + 1 / m2)
        out["fair_src"] = f"medián {len(prices[home])} kancelárií"
    return out


def basis_odds(o: dict) -> float:
    """Kurz, na ktorom sa meria value (rovnaký typ ako v backteste)."""
    b = config.VALUE_ODDS_BASIS
    if b == "max":
        return o["best"]
    if b == "pinnacle" and o.get("pinnacle"):
        return o["pinnacle"]
    return o["avg"]


def kelly_pct(p: float, o: float) -> float:
    f = config.KELLY_FRACTION * max(p * o - 1, 0) / (o - 1)
    return float(min(f, config.MAX_STAKE_PCT))


def predict_events(events: list[dict], sport: dict, book: EloBook, srv: ServeBook, model: Calibrated,
                   idx: FullNameIndex, threshold: float, now: dt.datetime,
                   w_blend: tuple[float, float] = blend.DEFAULT) -> tuple[list[dict], list[str]]:
    key = sport["key"]
    tour = tour_for(key)
    surface = surface_for(key, sport.get("title", ""))
    best_of = best_of_for(key)
    out, unmatched = [], []
    for ev in events:
        home, away = ev.get("home_team", ""), ev.get("away_team", "")
        if not home or not away or "/" in home or "/" in away:
            continue  # štvorhra
        start = pd.Timestamp(ev["commence_time"]).tz_convert("UTC")
        ka, kb = idx.match(home, tour), idx.match(away, tour)
        for nm, k in ((home, ka), (away, kb)):
            if k is None:
                unmatched.append(f"{tour}: {nm}")
        odds = parse_odds(ev, home, away)
        row = {
            "id": ev.get("id"), "sport_key": key, "tournament": sport.get("title", key), "tour": tour,
            "surface": surface, "best_of": best_of, "start": start.isoformat(),
            "started": bool(start <= pd.Timestamp(now)),
            "players": [home, away], "keys": [ka, kb], "odds": odds,
        }
        if ka is None or kb is None or ka == kb:
            row["p"] = None
            out.append(row)
            continue
        snap = snapshot(book, ka, kb, surface, start.tz_localize(None))
        sa, sb = srv.get(ka), srv.get(kb)
        spw_a = srv.expected(sa, sb, tour, surface)
        spw_b = srv.expected(sb, sa, tour, surface)
        snap["a_spw"], snap["b_spw"] = spw_a, spw_b
        has_serve = min(sa.n, sb.n) >= config.MIN_SERVE_MATCHES
        snap["a_mk"] = float(markov.match_prob(round(spw_a, 3), round(spw_b, 3), best_of)) if has_serve else 0.5
        X = features(pd.DataFrame([snap]), best_of=best_of)
        p = float(model.proba(X)[0])
        row["p"] = [p, 1 - p]
        row["fair_odds"] = [1 / max(p, 1e-6), 1 / max(1 - p, 1e-6)]
        row["elo"] = [[round(snap["a_elo"]), round(snap["a_surf"])], [round(snap["b_elo"]), round(snap["b_surf"])]]
        row["spw"] = [round(spw_a, 3), round(spw_b, 3)] if has_serve else None
        row["n"] = [int(snap["a_n"]), int(snap["b_n"])]
        row["rank"] = [None if np.isnan(snap["a_rank"]) else int(snap["a_rank"]),
                       None if np.isnan(snap["b_rank"]) else int(snap["b_rank"])]
        row["value"] = None
        if odds:
            # === jadro stratégie v2: základ je TRH, model ho len jemne opraví ===
            fair_home = odds.get("fair_home")
            if fair_home is None:
                ia, ib = 1 / odds[home]["avg"], 1 / odds[away]["avg"]
                fair_home = ia / (ia + ib)
                odds["fair_src"] = "priemer kancelárií"
            row["market_p"] = [fair_home, 1 - fair_home]
            pf = float(blend.apply(w_blend, np.array([p]), np.array([fair_home]))[0])
            row["p_final"] = [pf, 1 - pf]
            row["blend"] = {"w_market": w_blend[0], "w_model": w_blend[1], "src": odds.get("fair_src")}
            edges = []
            for i, side in enumerate((home, away)):
                o = basis_odds(odds[side])
                edges.append(row["p_final"][i] * o - 1)
                odds[side]["edge"] = edges[-1]
                odds[side]["edge_best"] = row["p_final"][i] * odds[side]["best"] - 1
            i = int(np.argmax(edges))
            side = (home, away)[i]
            o_basis = basis_odds(odds[side])
            reliable = min(row["n"]) >= MIN_MATCHES_FOR_VALUE
            row["reliable"] = reliable
            p_market = fair_home if i == 0 else 1 - fair_home
            p_pick = row["p_final"][i]
            disagree = abs(p_pick - p_market)
            reasons = []
            if odds[side]["n_books"] < 3:
                reasons.append("málo kancelárií na porovnanie")
            if edges[i] < threshold:
                reasons.append("malá výhoda")
            if edges[i] > config.MAX_EDGE:
                reasons.append("príliš veľká nezhoda s trhom")
            if disagree > config.MAX_MARKET_DISAGREEMENT:
                reasons.append(f"model sa líši od trhu o {disagree * 100:.0f} p. b.")
            if not (config.MIN_ODDS <= o_basis <= config.MAX_ODDS):
                reasons.append("kurz mimo rozsahu")
            if not reliable:
                reasons.append("málo zápasov v histórii")
            if row["started"]:
                reasons.append("zápas už začal")
            row["skip_reasons"] = reasons
            if not reasons:
                row["value"] = {
                    "side": i, "player": side, "p": p_pick, "p_model": row["p"][i], "p_market": p_market,
                    "odds": o_basis,
                    "best_odds": odds[side]["best"], "best_book": odds[side]["best_book"],
                    "edge": edges[i], "kelly_pct": kelly_pct(p_pick, odds[side]["best"]),
                    "min_odds": (1 + threshold) / p_pick, "disagree": disagree,
                }
        out.append(row)
    return out, unmatched
