"""Vlastná história kurzov: každé stiahnutie kurzov z The Odds API sa uloží (state/odds/RRRR-MM.csv).

Z toho sa počíta:
  * záverečný kurz (posledný záznam pred začiatkom zápasu) -> CLV papierových stávok,
  * vlastný backtest proti kurzom, keď historické kurzy z tennis-data.co.uk nie sú k dispozícii.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import config
from src.predict import parse_odds

COLS = ["snap_at", "kind", "event_id", "sport_key", "tournament", "tour", "surface", "best_of", "start",
        "a", "b", "key_a", "key_b", "p_a", "n_books", "avg_a", "avg_b", "best_a", "best_b", "book_a", "book_b",
        "pin_a", "pin_b"]


def _dir() -> str:
    return os.path.join(config.STATE_DIR, "odds")


def rows_from_predictions(preds: list[dict], snap_at: pd.Timestamp, kind: str = "main") -> list[dict]:
    rows = []
    for m in preds:
        o = m.get("odds") or {}
        a, b = m["players"]
        if a not in o or b not in o:
            continue
        rows.append({
            "snap_at": snap_at.isoformat(), "kind": kind, "event_id": m["id"], "sport_key": m["sport_key"],
            "tournament": m["tournament"], "tour": m["tour"], "surface": m["surface"], "best_of": m["best_of"],
            "start": m["start"], "a": a, "b": b, "key_a": m["keys"][0], "key_b": m["keys"][1],
            "p_a": (m["p"][0] if m.get("p") else np.nan), "n_books": o[a]["n_books"],
            "avg_a": o[a]["avg"], "avg_b": o[b]["avg"], "best_a": o[a]["best"], "best_b": o[b]["best"],
            "book_a": o[a]["best_book"], "book_b": o[b]["best_book"],
            "pin_a": o[a].get("pinnacle"), "pin_b": o[b].get("pinnacle"),
        })
    return rows


def rows_from_events(events: list[dict], sport: dict, snap_at: pd.Timestamp, kind: str = "snap") -> list[dict]:
    """Bez modelu – len kurzy (rýchle priebežné snímky pred zápasmi)."""
    rows = []
    for ev in events:
        a, b = ev.get("home_team", ""), ev.get("away_team", "")
        if not a or not b or "/" in a or "/" in b:
            continue
        o = parse_odds(ev, a, b)
        if not o:
            continue
        rows.append({
            "snap_at": snap_at.isoformat(), "kind": kind, "event_id": ev.get("id"), "sport_key": sport["key"],
            "tournament": sport.get("title", sport["key"]), "tour": None, "surface": None, "best_of": None,
            "start": pd.Timestamp(ev["commence_time"]).tz_convert("UTC").isoformat(), "a": a, "b": b,
            "key_a": None, "key_b": None, "p_a": np.nan, "n_books": o[a]["n_books"],
            "avg_a": o[a]["avg"], "avg_b": o[b]["avg"], "best_a": o[a]["best"], "best_b": o[b]["best"],
            "book_a": o[a]["best_book"], "book_b": o[b]["best_book"],
            "pin_a": o[a].get("pinnacle"), "pin_b": o[b].get("pinnacle"),
        })
    return rows


def append(rows: list[dict]) -> int:
    if not rows:
        return 0
    os.makedirs(_dir(), exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    month = pd.Timestamp(rows[0]["snap_at"]).strftime("%Y-%m")
    path = os.path.join(_dir(), f"{month}.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return len(df)


def load() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(_dir(), "*.csv")))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["snap_at"] = pd.to_datetime(df["snap_at"], utc=True, format="ISO8601")
    df["start"] = pd.to_datetime(df["start"], utc=True, format="ISO8601")
    return df


def fair_a(row) -> float:
    """Férová pravdepodobnosť hráča A bez marže: Pinnacle, inak priemer kancelárií."""
    for x, y in ((row.get("pin_a"), row.get("pin_b")), (row.get("avg_a"), row.get("avg_b"))):
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if x > 1 and y > 1:
            return (1 / x) / (1 / x + 1 / y)
    return np.nan


def closing(snaps: pd.DataFrame) -> pd.DataFrame:
    """Posledná snímka každého zápasu pred jeho začiatkom."""
    s = snaps[snaps["snap_at"] < snaps["start"]]
    return s.sort_values("snap_at").groupby("event_id").tail(1).set_index("event_id")


def update_bet_closing(log: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Doplní k papierovým stávkam záverečný kurz a CLV."""
    if log.empty or snaps.empty:
        return log
    close = closing(snaps)
    for c in ("close_at",):
        log[c] = log[c].astype(object)
    for i, r in log.iterrows():
        eid = str(r["bet_id"]).split("|")[0]
        if eid not in close.index:
            continue
        c = close.loc[eid]
        created = pd.Timestamp(r["created_at"])
        if c["snap_at"] <= created + pd.Timedelta(minutes=30):
            continue  # žiadna novšia snímka od tipu – záverečný kurz zatiaľ nepoznáme
        side_a = r["player"] == c["a"]
        if not side_a and r["player"] != c["b"]:
            continue
        basis = r.get("odds_basis") or config.VALUE_ODDS_BASIS
        col = {"avg": "avg", "max": "best", "pinnacle": "pin"}.get(basis, "avg")
        close_odds = c[f"{col}_a" if side_a else f"{col}_b"]
        if pd.isna(close_odds):
            close_odds = c["avg_a" if side_a else "avg_b"]
        fa = fair_a(c)
        fair_p = fa if side_a else 1 - fa
        log.at[i, "close_odds"] = round(float(close_odds), 3)
        log.at[i, "close_fair_p"] = round(float(fair_p), 4) if not pd.isna(fair_p) else np.nan
        log.at[i, "close_at"] = c["snap_at"].isoformat()
        log.at[i, "clv"] = round(float(r["odds"]) / float(close_odds) - 1, 4)
        log.at[i, "clv_ev"] = round(float(r["odds"]) * float(fair_p) - 1, 4) if not pd.isna(fair_p) else np.nan
    return log


def evaluate(snaps: pd.DataFrame, hist: pd.DataFrame, threshold: float) -> dict:
    """Vlastný backtest: ranné kurzy + predikcia modelu vs. skutočný výsledok."""
    out = {"snapshots": int(len(snaps)), "events": int(snaps["event_id"].nunique()) if len(snaps) else 0}
    if snaps.empty:
        return out
    main = snaps[snaps["p_a"].notna() & snaps["key_a"].notna() & snaps["key_b"].notna()]
    main = main.sort_values("snap_at").groupby("event_id").head(1)
    if main.empty:
        return out
    close = closing(snaps)
    rows = []
    for r in main.itertuples(index=False):
        start = r.start.tz_convert(None)
        win = (hist["start"] >= start - pd.Timedelta(days=16)) & (hist["start"] <= start + pd.Timedelta(days=1))
        m = hist[win & (((hist["w_key"] == r.key_a) & (hist["l_key"] == r.key_b)) |
                        ((hist["w_key"] == r.key_b) & (hist["l_key"] == r.key_a)))]
        if m.empty or "walkover" in str(m.iloc[0]["comment"]).lower():
            continue
        a_won = m.iloc[0]["w_key"] == r.key_a
        d = r._asdict()
        fo = fair_a(d)
        c = close.loc[r.event_id] if r.event_id in close.index else None
        fc = fair_a(c) if c is not None else np.nan
        rows.append({"a_won": a_won, "p": r.p_a, "fair_open": fo, "fair_close": fc,
                     "avg_a": r.avg_a, "avg_b": r.avg_b, "start": r.start})
    ev = pd.DataFrame(rows)
    out["settled_events"] = int(len(ev))
    if ev.empty:
        return out
    y = ev["a_won"].astype(float).values

    def metrics(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        ok = ~np.isnan(p)
        if ok.sum() == 0:
            return None
        pp, yy = p[ok], y[ok]
        return {"n": int(ok.sum()), "accuracy": float(((pp > 0.5) == (yy == 1)).mean()),
                "log_loss": float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())}

    out["model"] = metrics(ev["p"])
    out["market_open"] = metrics(ev["fair_open"])
    out["market_close"] = metrics(ev["fair_close"])
    # stratégia value tipov na ranných priemerných kurzoch
    ea = ev["p"] * ev["avg_a"] - 1
    eb = (1 - ev["p"]) * ev["avg_b"] - 1
    pick_a = ea >= eb
    edge = np.where(pick_a, ea, eb)
    odds = np.where(pick_a, ev["avg_a"], ev["avg_b"])
    won = np.where(pick_a, ev["a_won"], ~ev["a_won"].astype(bool))
    sel = (edge >= threshold) & (odds >= config.MIN_ODDS) & (odds <= config.MAX_ODDS)
    prof = np.where(won, odds - 1, -1.0)[sel]
    out["strategy"] = {"bets": int(sel.sum()), "roi": float(prof.mean()) if sel.sum() else None,
                       "profit": float(prof.sum())}
    # pohyb trhu k modelu: posunul sa kurz od rána smerom k názoru modelu?
    mv = ev.dropna(subset=["fair_open", "fair_close"])
    if len(mv):
        toward = np.sign(mv["p"] - mv["fair_open"]) == np.sign(mv["fair_close"] - mv["fair_open"])
        moved = (mv["fair_close"] - mv["fair_open"]).abs() > 0.005
        out["market_moves_toward_model"] = float(toward[moved].mean()) if moved.sum() else None
        out["market_moved_n"] = int(moved.sum())
    return out
