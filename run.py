"""Tenisový predictor – hlavný skript.

    python run.py update          # všetko: dáta -> Elo -> backtest -> dnešné tipy -> dashboard
    python run.py update --offline  # bez sťahovania (použije dáta v data/raw)
    python run.py backtest        # len história a backtest
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import config
from src import backtest, elo, tdata, tml, tracker
from src.names import FullNameIndex
from src.odds_api import OddsAPI, QuotaLow
from src.predict import player_index, predict_events

ALIASES = os.path.join(config.STATE_DIR, "aliases.csv")


def _clean(o):
    """JSON bez NaN/Inf a numpy typov."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    return o


def write_json(name: str, obj):
    os.makedirs(config.DOCS_DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DOCS_DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, separators=(",", ":"))


def players_from_hist(hist: pd.DataFrame) -> dict:
    w = hist[["w_key", "winner", "date"]].set_axis(["key", "name", "date"], axis=1)
    l = hist[["l_key", "loser", "date"]].set_axis(["key", "name", "date"], axis=1)
    a = pd.concat([w, l]).sort_values("date")
    g = a.groupby("key")
    info = pd.DataFrame({"name": g["name"].last(), "last": g["date"].max(), "n": g.size()})
    return {k: {"name": r.name, "last": r.last, "n": int(r.n)} for k, r in zip(info.index, info.itertuples())}


def ratings_export(book: elo.EloBook, model, last_date: pd.Timestamp) -> dict:
    players = []
    cutoff = last_date - pd.Timedelta(days=548)
    for key, p in book.p.items():
        if p.last is None or p.last < cutoff or p.n < 10:
            continue
        players.append({
            "key": key, "name": p.name, "tour": key.split("|")[0], "elo": round(p.elo, 1),
            "surf": {s: round(v, 1) for s, v in p.surf.items()}, "n": p.n, "sn": p.surf_n,
            "rank": None if math.isnan(p.rank) else int(p.rank), "last": str(p.last.date()),
        })
    players.sort(key=lambda x: -x["elo"])
    return {"model": model.to_json(), "players": players, "data_until": str(last_date.date())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "backtest"])
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")

    print("1) Historické dáta")
    if not args.offline:
        tml.download()
        tdata.download()
    hist = tml.load_all()
    last_date = hist["date"].max()
    print(f"   {len(hist)} zápasov (TennisMyLife), posledný {last_date.date()}")
    n_odds = tdata.attach_odds(hist, tdata.load_all(), FullNameIndex(players_from_hist(hist)))
    print(f"   kurzy z tennis-data.co.uk pripojené k {n_odds} zápasom")

    print("2) Elo ratingy")
    feats, book = elo.run(hist)

    print("3) Backtest (walk-forward)")
    bt, _ = backtest.run(hist, feats)
    bt["generated_at"] = now.isoformat()
    bt["data_until"] = str(last_date.date())
    write_json("backtest.json", bt)
    ch = bt["chosen"]
    print(f"   prah edge {bt['chosen_edge']:.3f} | tréning ROI {ch['train']['roi']} ({ch['train']['bets']} stávok)"
          f" | test ROI {ch['test']['roi']} ({ch['test']['bets']} stávok) | verdikt: {bt['verdict']}")

    model = backtest.fit_latest(hist, feats)
    write_json("ratings.json", ratings_export(book, model, last_date))
    if args.cmd == "backtest":
        return

    print("4) Dnešné zápasy (The Odds API)")
    api = OddsAPI()
    preds, unmatched, notes = [], [], []
    threshold = bt["chosen_edge"]
    if not api.enabled:
        notes.append("Chýba ODDS_API_KEY – predikcie nadchádzajúcich zápasov sa preskočili.")
    else:
        idx = player_index(book, ALIASES)
        try:
            sports = api.tennis_sports()
            print(f"   aktívne turnaje: {[s['key'] for s in sports]}")
            for sp in sports:
                try:
                    events = api.odds(sp["key"])
                except QuotaLow as e:
                    notes.append(str(e))
                    break
                p, u = predict_events(events, sp, book, model, idx, threshold, now.to_pydatetime())
                preds += p
                unmatched += u
        except Exception as e:
            notes.append(f"Chyba pri The Odds API: {e}")
        notes += api.log
    preds.sort(key=lambda m: m["start"])

    print("5) Papierové stávky")
    log = tracker.load()
    log, n_new = tracker.add_value_bets(log, preds, now)
    try:
        log = tracker.settle_with_scores(log, api, now)
    except Exception as e:
        notes.append(f"Vyhodnotenie cez /scores zlyhalo: {e}")
    log = tracker.settle_with_history(log, hist, now)
    tracker.save(log)
    print(f"   nové tipy: {n_new}, čakajúce: {(log['status'] == 'pending').sum()}")

    write_json("predictions.json", {
        "generated_at": now.isoformat(), "threshold": threshold, "basis": config.VALUE_ODDS_BASIS,
        "min_odds": config.MIN_ODDS, "max_odds": config.MAX_ODDS, "quota_remaining": api.remaining,
        "matches": preds, "unmatched": sorted(set(unmatched)), "notes": notes, "verdict": bt["verdict"],
    })
    write_json("tracker.json", tracker.summary(log))
    print("Hotovo.")


if __name__ == "__main__":
    sys.exit(main())
