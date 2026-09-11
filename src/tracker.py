"""Papierové stávky (forward test): každý value tip sa zapíše s kurzom v čase tipu a neskôr vyhodnotí.
Toto je jediný skutočne poctivý test – model tu nevidí budúcnosť ani náhodou."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config

COLS = ["bet_id", "created_at", "start", "sport_key", "tournament", "tour", "surface", "player", "opponent",
        "player_key", "opponent_key", "p_model", "odds_basis", "odds", "best_odds", "best_book", "edge",
        "kelly_pct", "stake", "status", "settled_at", "profit", "note"]


def path() -> str:
    return os.path.join(config.STATE_DIR, "bet_log.csv")


def load() -> pd.DataFrame:
    if os.path.exists(path()):
        df = pd.read_csv(path(), dtype={"bet_id": str})
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame):
    os.makedirs(config.STATE_DIR, exist_ok=True)
    df[COLS].to_csv(path(), index=False)


def add_value_bets(log: pd.DataFrame, preds: list[dict], now: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    have = set(log["bet_id"].astype(str))
    new = []
    for m in preds:
        v = m.get("value")
        if not v:
            continue
        bid = f"{m['id']}|{v['player']}"
        if bid in have:
            continue
        opp_i = 1 - v["side"]
        new.append({
            "bet_id": bid, "created_at": now.isoformat(), "start": m["start"], "sport_key": m["sport_key"],
            "tournament": m["tournament"], "tour": m["tour"], "surface": m["surface"],
            "player": v["player"], "opponent": m["players"][opp_i],
            "player_key": m["keys"][v["side"]], "opponent_key": m["keys"][opp_i],
            "p_model": round(v["p"], 4), "odds_basis": config.VALUE_ODDS_BASIS, "odds": round(v["odds"], 3),
            "best_odds": v["best_odds"], "best_book": v["best_book"], "edge": round(v["edge"], 4),
            "kelly_pct": round(v["kelly_pct"], 4), "stake": 1.0, "status": "pending",
            "settled_at": "", "profit": np.nan, "note": "",
        })
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    return log, len(new)


def _settle_row(log, i, won: bool | None, now, note=""):
    if won is None:
        log.at[i, "status"], log.at[i, "profit"] = "void", 0.0
    else:
        log.at[i, "status"] = "won" if won else "lost"
        log.at[i, "profit"] = (float(log.at[i, "odds"]) - 1) if won else -1.0
    log.at[i, "settled_at"] = now.isoformat()
    log.at[i, "note"] = note


def settle_with_scores(log: pd.DataFrame, api, now: pd.Timestamp) -> pd.DataFrame:
    pend = log[(log["status"] == "pending") & (pd.to_datetime(log["start"], utc=True) < now - pd.Timedelta(hours=3))]
    if pend.empty or not api or not api.enabled or not config.SETTLE_WITH_SCORES:
        return log
    for sport_key in pend["sport_key"].unique():
        try:
            events = api.scores(sport_key, days_from=3)
        except Exception as e:  # nedostatok kreditov a pod.
            print(f"  ! scores {sport_key}: {e}")
            continue
        by_id = {e["id"]: e for e in events}
        for i in pend[pend["sport_key"] == sport_key].index:
            ev = by_id.get(str(log.at[i, "bet_id"]).split("|")[0])
            if not ev or not ev.get("completed"):
                continue
            sc = {s["name"]: s.get("score") for s in (ev.get("scores") or [])}
            try:
                a, b = float(sc[log.at[i, "player"]]), float(sc[log.at[i, "opponent"]])
            except (KeyError, TypeError, ValueError):
                continue
            if a == b:
                _settle_row(log, i, None, now, "nejasný výsledok")
            else:
                _settle_row(log, i, a > b, now, "the-odds-api")
    return log


def settle_with_history(log: pd.DataFrame, hist: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Záloha: vyhodnotenie z histórie výsledkov (TennisMyLife)."""
    pend = log[log["status"] == "pending"]
    if pend.empty:
        return log
    for i, r in pend.iterrows():
        start = pd.Timestamp(r["start"]).tz_convert(None) if pd.Timestamp(r["start"]).tzinfo else pd.Timestamp(r["start"])
        if start > now.tz_convert(None) - pd.Timedelta(days=1):
            continue
        # v TennisMyLife je dátum začiatok turnaja (+ odhad podľa kola), preto širšie okno
        win = (hist["start"] >= start - pd.Timedelta(days=16)) & (hist["start"] <= start + pd.Timedelta(days=1))
        pk, ok = r["player_key"], r["opponent_key"]
        m = hist[win & (((hist["w_key"] == pk) & (hist["l_key"] == ok)) | ((hist["w_key"] == ok) & (hist["l_key"] == pk)))]
        if len(m):
            row = m.iloc[0]
            c = str(row["comment"]).lower()
            if "walkover" in c or "w/o" in c:
                _settle_row(log, i, None, now, "walkover")
            else:
                _settle_row(log, i, row["w_key"] == pk, now, "výsledky" + (" (skreč)" if "retired" in c else ""))
        elif start < now.tz_convert(None) - pd.Timedelta(days=21):
            _settle_row(log, i, None, now, "výsledok sa nenašiel")
    return log


def summary(log: pd.DataFrame) -> dict:
    s = log[log["status"].isin(["won", "lost"])].copy()
    s = s.sort_values("start")
    out = {
        "total": int(len(log)), "pending": int((log["status"] == "pending").sum()),
        "settled": int(len(s)), "won": int((s["status"] == "won").sum()),
        "profit": float(s["profit"].sum()) if len(s) else 0.0,
        "roi": float(s["profit"].sum() / len(s)) if len(s) else None,
        "avg_odds": float(s["odds"].mean()) if len(s) else None,
        "curve": [{"date": str(r.start)[:10], "cum": round(float(c), 2)} for r, c in zip(s.itertuples(), s["profit"].cumsum())],
        "recent": log.sort_values("start", ascending=False).head(60).replace({np.nan: None}).to_dict("records"),
    }
    return out
