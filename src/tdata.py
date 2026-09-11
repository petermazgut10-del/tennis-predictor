"""Historické kurzy z tennis-data.co.uk (voliteľné – slúžia na backtest proti stávkovým kanceláriám).

Výsledky na výpočet ratingov idú z TennisMyLife (src/tml.py). Tu sa len sťahujú kurzy a pripájajú
k zápasom. Keď stránka nie je dostupná, predictor funguje ďalej, len backtest nemá s čím porovnať.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

import config

BASE = "http://www.tennis-data.co.uk"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "http://www.tennis-data.co.uk/alldata.php",
}
ODDS_PAIRS = {"avg": ("AvgW", "AvgL"), "max": ("MaxW", "MaxL"), "pinnacle": ("PSW", "PSL"), "b365": ("B365W", "B365L")}


def _dir() -> str:
    return os.path.join(config.RAW_DIR, "odds")


def _path(tour: str, year: int, ext: str) -> str:
    return os.path.join(_dir(), f"{tour}_{year}{ext}")


def _get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=(10, 30))
    ok = r.status_code == 200 and len(r.content) > 5000 and not r.content.lstrip()[:15].lower().startswith(b"<")
    return ok, r


def download(verbose: bool = True) -> None:
    os.makedirs(_dir(), exist_ok=True)
    this_year = dt.date.today().year
    # rýchla skúška dostupnosti, aby sme pri výpadku nečakali na desiatky timeoutov
    try:
        ok, r = _get(f"{BASE}/{this_year - 1}/{this_year - 1}.xlsx")
    except requests.RequestException as e:
        ok, r = False, None
        if verbose:
            print(f"   tennis-data.co.uk nedostupné ({e.__class__.__name__}) – backtest použije uložené kurzy, ak sú")
        return
    if not ok:
        if verbose:
            print(f"   tennis-data.co.uk nedostupné (HTTP {r.status_code}) – backtest použije uložené kurzy, ak sú")
        return
    got = 0
    t0 = time.time()
    for tour in config.TOURS:
        if time.time() - t0 > config.ODDS_DOWNLOAD_BUDGET_S:
            break
        for year in range(config.ODDS_START_YEAR, this_year + 1):
            if time.time() - t0 > config.ODDS_DOWNLOAD_BUDGET_S:
                print(f"   tennis-data.co.uk: časový limit {config.ODDS_DOWNLOAD_BUDGET_S} s – zvyšok stiahne ďalší beh")
                break
            existing = [p for p in (_path(tour, year, ".xlsx"), _path(tour, year, ".xls")) if os.path.exists(p)]
            if existing and year < this_year - 1:
                continue
            folder = f"{year}" if tour == "ATP" else f"{year}w"
            for ext in ("xlsx", "xls"):
                try:
                    ok, r = _get(f"{BASE}/{folder}/{year}.{ext}")
                except requests.RequestException:
                    ok = False
                if ok:
                    with open(_path(tour, year, "." + ext), "wb") as f:
                        f.write(r.content)
                    got += 1
                    break
                time.sleep(0.3)
    if verbose:
        print(f"   tennis-data.co.uk: stiahnuté {got} súborov s kurzami")


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def load_file(path: str, tour: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    d = pd.DataFrame(index=raw.index)
    d["tour"] = tour
    d["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    d["winner"] = raw["Winner"].astype(str).str.strip()
    d["loser"] = raw["Loser"].astype(str).str.strip()
    for name, (cw, cl) in ODDS_PAIRS.items():
        d[f"odds_w_{name}"] = _to_num(raw[cw]) if cw in raw.columns else np.nan
        d[f"odds_l_{name}"] = _to_num(raw[cl]) if cl in raw.columns else np.nan
    return d


def load_all() -> pd.DataFrame | None:
    if not os.path.isdir(_dir()):
        return None
    frames = []
    for f in sorted(os.listdir(_dir())):
        if f.endswith((".xlsx", ".xls")):
            try:
                frames.append(load_file(os.path.join(_dir(), f), f.split("_")[0]))
            except Exception as e:
                print(f"  ! nepodarilo sa načítať {f}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    for c in [c for c in df.columns if c.startswith("odds_")]:
        df.loc[(df[c] < 1.001) | (df[c] > 200), c] = np.nan
    return df


def attach_odds(hist: pd.DataFrame, odds: pd.DataFrame | None, idx) -> int:
    """Pripojí kurzy k zápasom v hist (na mieste). idx = FullNameIndex nad hráčmi z hist. Vráti počet zhôd."""
    if odds is None or odds.empty:
        return 0
    cache = {}

    def key(tour, name):
        k = (tour, name)
        if k not in cache:
            cache[k] = idx.match_td(name, tour)
        return cache[k]

    o = odds.copy()
    o["w_key"] = [key(t, n) for t, n in zip(o["tour"], o["winner"])]
    o["l_key"] = [key(t, n) for t, n in zip(o["tour"], o["loser"])]
    o = o.dropna(subset=["w_key", "l_key"])
    o = o.rename(columns={"date": "odds_date"})
    h = hist[["w_key", "l_key", "date", "start"]].reset_index().rename(columns={"index": "hid"})
    m = h.merge(o[["w_key", "l_key", "odds_date"] + [c for c in o.columns if c.startswith("odds_") and c != "odds_date"]],
                on=["w_key", "l_key"])
    # dátum z tennis-data je deň zápasu; v TennisMyLife je začiatok turnaja (+ odhad podľa kola)
    m = m[(m["odds_date"] >= m["start"] - pd.Timedelta(days=3)) & (m["odds_date"] <= m["start"] + pd.Timedelta(days=16))]
    m["gap"] = (m["odds_date"] - m["date"]).abs()
    m = m.sort_values("gap").drop_duplicates("hid")
    for c in [c for c in m.columns if c.startswith("odds_") and c != "odds_date"]:
        hist.loc[m["hid"].values, c] = m[c].values
    return int(len(m))
