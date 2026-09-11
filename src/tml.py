"""Výsledky zápasov z TennisMyLife (https://stats.tennismylife.org, licencia MIT).

Denne aktualizované ATP, Challenger a WTA zápasy vrátane práve prebiehajúcich turnajov.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

import config

BASE = "https://stats.tennismylife.org/data"
HEADERS = {"User-Agent": "tennis-predictor (personal research; github.com)"}

# približný posun dňa v turnaji podľa kola (poradie zápasov pre Elo a "dni od posledného zápasu")
ROUND_DAY = {"Q1": -3, "Q2": -2, "Q3": -1, "RR": 1, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6,
             "BR": 7, "F": 7}
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "RR": 3, "R128": 4, "R64": 5, "R32": 6, "R16": 7, "QF": 8, "SF": 9,
               "BR": 10, "F": 11}


def files_for(year: int) -> list[tuple[str, str]]:
    """(tour, názov súboru) pre daný rok."""
    out = [("ATP", f"{year}.csv"), ("WTA", f"{year}_wta.csv")]
    if config.INCLUDE_CHALLENGERS:
        out.append(("ATP", f"{year}_challenger.csv"))
    return [(t, f) for t, f in out if t in config.TOURS]


ONGOING = [("ATP", "ongoing_tourneys.csv"), ("WTA", "wta_ongoing_tourneys.csv"),
           ("ATP", "challenger_ongoing_tourneys.csv")]


def _local(name: str) -> str:
    return os.path.join(config.RAW_DIR, "tml", name)


def download(verbose: bool = True) -> None:
    os.makedirs(os.path.join(config.RAW_DIR, "tml"), exist_ok=True)
    this_year = dt.date.today().year
    todo = []
    for year in range(config.START_YEAR, this_year + 1):
        for tour, name in files_for(year):
            # staršie roky sa menia zriedka – sťahujú sa, len keď chýbajú alebo sú staršie ako 30 dní
            p = _local(name)
            fresh = os.path.exists(p) and (time.time() - os.path.getmtime(p) < 30 * 86400)
            if year < this_year - 1 and fresh:
                continue
            todo.append(name)
    todo += [n for t, n in ONGOING if t in config.TOURS and (config.INCLUDE_CHALLENGERS or "challenger" not in n)]
    ok = 0
    for name in todo:
        try:
            r = requests.get(f"{BASE}/{name}", headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            print(f"  ! {name}: {e}")
            continue
        if r.status_code == 200 and r.content[:10].startswith(b"tourney_id"):
            with open(_local(name), "wb") as f:
                f.write(r.content)
            ok += 1
        else:
            print(f"  ! {name}: HTTP {r.status_code}, {len(r.content)} B")
    if verbose:
        print(f"   TennisMyLife: stiahnuté {ok}/{len(todo)} súborov")


def _read(path: str, tour: str, ongoing: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if raw.empty:
        return pd.DataFrame()
    d = pd.DataFrame(index=raw.index)
    d["tour"] = tour
    d["tourney_id"] = raw["tourney_id"]
    d["tournament"] = raw["tourney_name"]
    d["level"] = raw["tourney_level"]
    d["challenger"] = "challenger" in os.path.basename(path)
    start = pd.to_datetime(raw["tourney_date"], format="%Y%m%d", errors="coerce")
    rnd = raw["round"].str.strip()
    gs = raw["tourney_level"].eq("G")
    offset = rnd.map(ROUND_DAY).fillna(3) * np.where(gs, 2, 1)
    # v "ongoing" súboroch je tourney_date už dátum zápasu
    d["date"] = start if ongoing else start + pd.to_timedelta(offset, unit="D")
    d["start"] = start
    d["round"] = rnd
    d["round_order"] = rnd.map(ROUND_ORDER).fillna(5)
    d["match_num"] = pd.to_numeric(raw["match_num"], errors="coerce").fillna(0)
    d["surface"] = raw["surface"].str.strip().str.title().replace({"Carpet": "Hard", "": "Hard"})
    d["best_of"] = pd.to_numeric(raw["best_of"], errors="coerce").fillna(3).astype(int)
    d["winner"] = raw["winner_name"].str.strip()
    d["loser"] = raw["loser_name"].str.strip()
    d["w_key"] = tour + "|" + raw["winner_id"].str.strip()
    d["l_key"] = tour + "|" + raw["loser_id"].str.strip()
    d["w_rank"] = pd.to_numeric(raw["winner_rank"], errors="coerce")
    d["l_rank"] = pd.to_numeric(raw["loser_rank"], errors="coerce")
    score = raw["score"].str.upper()
    d["score"] = raw["score"]
    d["comment"] = np.where(score.str.contains("W/O|WALKOVER|DEF", regex=True), "Walkover",
                            np.where(score.str.contains("RET|ABN|ABD", regex=True), "Retired", "Completed"))
    return d


def load_all() -> pd.DataFrame:
    frames = []
    this_year = dt.date.today().year
    for year in range(config.START_YEAR, this_year + 1):
        for tour, name in files_for(year):
            p = _local(name)
            if os.path.exists(p):
                frames.append(_read(p, tour, ongoing=False))
    for tour, name in ONGOING:
        p = _local(name)
        if tour in config.TOURS and os.path.exists(p) and (config.INCLUDE_CHALLENGERS or "challenger" not in name):
            frames.append(_read(p, tour, ongoing=True))
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("Chýbajú historické dáta – sťahovanie z TennisMyLife zlyhalo, pozri log vyššie.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date"])
    df = df[(df["winner"] != "") & (df["loser"] != "") & (df["w_key"] != df["l_key"])]
    df = df[~df["w_key"].str.endswith("|") & ~df["l_key"].str.endswith("|")]
    df = df[df["surface"].isin(["Hard", "Clay", "Grass"])]
    # zápas môže byť v "ongoing" aj v ročnom súbore – necháme jeden
    df["_pair"] = df[["w_key", "l_key"]].min(axis=1) + df[["w_key", "l_key"]].max(axis=1)
    df = df.drop_duplicates(subset=["tourney_id", "_pair", "round"], keep="first").drop(columns="_pair")
    for name in ("avg", "max", "pinnacle", "b365"):
        df[f"odds_w_{name}"] = np.nan
        df[f"odds_l_{name}"] = np.nan
    df = df.sort_values(["date", "start", "round_order", "match_num"], kind="stable").reset_index(drop=True)
    return df
