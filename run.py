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
from src import backtest, elo, markov, odds_store, serve, tdata, tml, tracker
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


def add_markov(feats: pd.DataFrame, srv_feats: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Doplní k Elo príznakom očakávané podanie oboch hráčov a z neho šancu podľa modelu bodov."""
    f = feats.join(srv_feats.loc[feats.index])
    ok = (f["a_ns"] >= config.MIN_SERVE_MATCHES) & (f["b_ns"] >= config.MIN_SERVE_MATCHES)
    mk = np.full(len(f), 0.5)
    if ok.any():
        mk[ok.values] = markov.match_prob_vec(f.loc[ok, "a_spw"], f.loc[ok, "b_spw"],
                                              hist.loc[f.index[ok], "best_of"])
    f["a_mk"] = mk
    f["b_mk"] = 1 - mk
    return f


def players_from_hist(hist: pd.DataFrame) -> dict:
    w = hist[["w_key", "winner", "date"]].set_axis(["key", "name", "date"], axis=1)
    l = hist[["l_key", "loser", "date"]].set_axis(["key", "name", "date"], axis=1)
    a = pd.concat([w, l]).sort_values("date")
    g = a.groupby("key")
    info = pd.DataFrame({"name": g["name"].last(), "last": g["date"].max(), "n": g.size()})
    return {k: {"name": r.name, "last": r.last, "n": int(r.n)} for k, r in zip(info.index, info.itertuples())}


def ratings_export(book: elo.EloBook, srv: serve.ServeBook, model, last_date: pd.Timestamp) -> dict:
    players = []
    cutoff = last_date - pd.Timedelta(days=548)
    for key, p in book.p.items():
        if p.last is None or p.last < cutoff or p.n < 10:
            continue
        sp = srv.p.get(key)
        players.append({
            "key": key, "name": p.name, "tour": key.split("|")[0], "elo": round(p.elo, 1),
            "surf": {s: round(v, 1) for s, v in p.surf.items()}, "n": p.n, "sn": p.surf_n,
            "rank": None if math.isnan(p.rank) else int(p.rank), "last": str(p.last.date()),
            "s": round(sp.s, 4) if sp else 0.0, "r": round(sp.r, 4) if sp else 0.0, "ns": sp.n if sp else 0,
        })
    players.sort(key=lambda x: -x["elo"])
    return {"model": model.to_json(), "players": players, "data_until": str(last_date.date()),
            "serve_base": {f"{t}|{s}": round(v, 4) for (t, s), v in srv.base.items()},
            "min_serve_matches": config.MIN_SERVE_MATCHES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "backtest", "snapshot"])
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    if args.cmd == "snapshot":
        return snapshot(now)

    print("1) Historické dáta")
    if not args.offline:
        tml.download()
        tdata.download()
    hist = tml.load_all()
    last_date = hist["date"].max()
    print(f"   {len(hist)} zápasov (TennisMyLife), posledný {last_date.date()}")
    n_odds = tdata.attach_odds(hist, tdata.load_all(), FullNameIndex(players_from_hist(hist)))
    print(f"   kurzy z tennis-data.co.uk pripojené k {n_odds} zápasom")

    print("2) Elo ratingy a sila podania")
    feats, book = elo.run(hist)
    srv_feats, srv = serve.run(hist)
    feats = add_markov(feats, srv_feats, hist)
    print(f"   hráči so štatistikou podania: {sum(1 for p in srv.p.values() if p.n >= 5)}")

    print("3) Backtest (walk-forward)")
    bt, _ = backtest.run(hist, feats)
    bt["generated_at"] = now.isoformat()
    bt["data_until"] = str(last_date.date())
    write_json("backtest.json", bt)
    ch = bt["chosen"]
    print(f"   prah edge {bt['chosen_edge']:.3f} | tréning ROI {ch['train']['roi']} ({ch['train']['bets']} stávok)"
          f" | test ROI {ch['test']['roi']} ({ch['test']['bets']} stávok) | verdikt: {bt['verdict']}")

    model = backtest.fit_latest(hist, feats)
    write_json("ratings.json", ratings_export(book, srv, model, last_date))
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
                p, u = predict_events(events, sp, book, srv, model, idx, threshold, now.to_pydatetime())
                preds += p
                unmatched += u
        except Exception as e:
            notes.append(f"Chyba pri The Odds API: {e}")
        notes += api.log
    preds.sort(key=lambda m: m["start"])

    print("5) Papierové stávky a história kurzov")
    n_snap = odds_store.append(odds_store.rows_from_predictions(preds, now, "main"))
    log = tracker.load()
    log, n_new = tracker.add_value_bets(log, preds, now)
    snaps = odds_store.load()
    log = odds_store.update_bet_closing(log, snaps)
    try:
        log = tracker.settle_with_scores(log, api, now)
    except Exception as e:
        notes.append(f"Vyhodnotenie cez /scores zlyhalo: {e}")
    log = tracker.settle_with_history(log, hist, now)
    tracker.save(log)
    print(f"   nové tipy: {n_new}, čakajúce: {(log['status'] == 'pending').sum()}, uložené kurzy: {n_snap}")
    hist_eval = odds_store.evaluate(snaps, hist, threshold)
    hist_eval["generated_at"] = now.isoformat()
    write_json("history.json", hist_eval)

    write_json("predictions.json", {
        "generated_at": now.isoformat(), "threshold": threshold, "basis": config.VALUE_ODDS_BASIS,
        "min_odds": config.MIN_ODDS, "max_odds": config.MAX_ODDS, "quota_remaining": api.remaining,
        "matches": preds, "unmatched": sorted(set(unmatched)), "notes": notes, "verdict": bt["verdict"],
    })
    write_json("tracker.json", tracker.summary(log))
    print("Hotovo.")


def snapshot(now: pd.Timestamp):
    """Rýchla snímka kurzov pred začiatkom papierových tipov (záverečný kurz -> CLV)."""
    log = tracker.load()
    if log.empty:
        print("Žiadne papierové tipy.")
        return
    st = pd.to_datetime(log["start"], utc=True, format="ISO8601")
    soon = log[(log["status"] == "pending") & (st > now + pd.Timedelta(minutes=10))
               & (st <= now + pd.Timedelta(hours=config.SNAPSHOT_WINDOW_H))]
    if soon.empty:
        print(f"Žiadny tip nezačína v najbližších {config.SNAPSHOT_WINDOW_H} h – nič netreba (0 kreditov).")
        return
    api = OddsAPI()
    if not api.enabled:
        print("Chýba ODDS_API_KEY.")
        return
    rows = []
    for key in soon["sport_key"].unique():
        try:
            events = api.odds(key)
        except QuotaLow as e:
            print(f"  ! {e}")
            break
        title = soon.loc[soon["sport_key"] == key, "tournament"].iloc[0]
        rows += odds_store.rows_from_events(events, {"key": key, "title": title}, now, "close")
    n = odds_store.append(rows)
    log = odds_store.update_bet_closing(log, odds_store.load())
    tracker.save(log)
    write_json("tracker.json", tracker.summary(log))
    print(f"Snímka kurzov: {n} zápasov, tipy pred začiatkom: {len(soon)}, zostáva kreditov: {api.remaining}")


if __name__ == "__main__":
    sys.exit(main())
