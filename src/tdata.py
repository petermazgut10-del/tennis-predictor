"""Sťahovanie a čistenie historických výsledkov + kurzov z tennis-data.co.uk."""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

import config
from src.names import canonical_map, td_key

BASES = ["http://www.tennis-data.co.uk", "https://www.tennis-data.co.uk"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "http://www.tennis-data.co.uk/alldata.php",
}


def _urls(tour: str, year: int) -> list[str]:
    folder = f"{year}" if tour == "ATP" else f"{year}w"
    return [f"{b}/{folder}/{year}.{ext}" for ext in ("xlsx", "xls") for b in BASES]


def _path(tour: str, year: int, ext: str) -> str:
    return os.path.join(config.RAW_DIR, f"{tour}_{year}{ext}")


def download(force_recent: bool = True, verbose: bool = True) -> list[str]:
    """Stiahne chýbajúce roky. Aktuálny a minulý rok sťahuje vždy znova (priebežne sa dopĺňajú)."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    this_year = dt.date.today().year
    got = []
    for tour in config.TOURS:
        for year in range(config.START_YEAR, this_year + 1):
            existing = [p for p in (_path(tour, year, ".xlsx"), _path(tour, year, ".xls")) if os.path.exists(p)]
            recent = year >= this_year - 1
            if existing and not (recent and force_recent):
                got.append(existing[0])
                continue
            ok = False
            for url in _urls(tour, year):
                try:
                    r = requests.get(url, headers=HEADERS, timeout=60)
                except requests.RequestException as e:
                    if verbose:
                        print(f"  ! {url}: {e}")
                    continue
                ctype = r.headers.get("content-type", "")
                if r.status_code == 200 and len(r.content) > 5000 and "html" not in ctype.lower():
                    ext = ".xlsx" if url.endswith(".xlsx") else ".xls"
                    with open(_path(tour, year, ext), "wb") as f:
                        f.write(r.content)
                    got.append(_path(tour, year, ext))
                    ok = True
                    if verbose:
                        print(f"  ✓ {tour} {year} ({len(r.content)//1024} kB)")
                    break
                if verbose:
                    print(f"  · {url}: HTTP {r.status_code}, {len(r.content)} B, {ctype}, {r.content[:80]!r}")
                time.sleep(0.3)
            if not ok:
                if existing:  # sťahovanie zlyhalo, použijeme starú kópiu
                    got.append(existing[0])
                elif verbose:
                    print(f"  – {tour} {year}: súbor nie je k dispozícii")
    return got


ODDS_PAIRS = {  # názov v našich dátach -> stĺpce (víťaz, porazený)
    "avg": ("AvgW", "AvgL"),
    "max": ("MaxW", "MaxL"),
    "pinnacle": ("PSW", "PSL"),
    "b365": ("B365W", "B365L"),
}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_file(path: str, tour: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    n = len(raw)

    def col(name, default=""):
        return raw[name] if name in raw.columns else pd.Series([default] * n)

    df = pd.DataFrame(index=raw.index)
    df["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    df["tournament"] = col("Tournament").astype(str).str.strip()
    df["location"] = col("Location").astype(str).str.strip()
    df["level"] = (col("Series") if "Series" in raw.columns else col("Tier")).astype(str).str.strip()
    df["court"] = col("Court").astype(str).str.strip()
    df["surface"] = raw["Surface"].astype(str).str.strip().str.title()
    df["round"] = col("Round").astype(str).str.strip()
    df["best_of"] = _to_num(col("Best of", 3)).fillna(3).astype(int)
    df["winner"] = raw["Winner"].astype(str).str.strip()
    df["loser"] = raw["Loser"].astype(str).str.strip()
    df["w_rank"] = _to_num(col("WRank", np.nan))
    df["l_rank"] = _to_num(col("LRank", np.nan))
    df["comment"] = col("Comment", "Completed").astype(str).str.strip()
    for name, (cw, cl) in ODDS_PAIRS.items():
        df[f"odds_w_{name}"] = _to_num(raw[cw]) if cw in raw.columns else np.nan
        df[f"odds_l_{name}"] = _to_num(raw[cl]) if cl in raw.columns else np.nan
    df["tour"] = tour
    df["row"] = np.arange(len(df))
    return df


def load_all(paths: list[str] | None = None) -> pd.DataFrame:
    if paths is None:
        paths = sorted(
            os.path.join(config.RAW_DIR, f) for f in os.listdir(config.RAW_DIR)
            if f.endswith((".xlsx", ".xls"))
        )
    if not paths:
        raise SystemExit("Chýbajú historické dáta (data/raw je prázdne) – sťahovanie z tennis-data.co.uk zlyhalo, pozri log vyššie.")
    frames = []
    for p in paths:
        tour = os.path.basename(p).split("_")[0]
        try:
            frames.append(load_file(p, tour))
        except Exception as e:  # jeden pokazený súbor nesmie zhodiť celý beh
            print(f"  ! nepodarilo sa načítať {p}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["date"])
    df = df[(df["winner"] != "") & (df["loser"] != "") & (df["winner"] != "nan")]
    df["surface"] = df["surface"].replace({"Carpet": "Hard"})  # carpet sa dnes nehrá, najbližší je hard (indoor)
    df = df[df["surface"].isin(["Hard", "Clay", "Grass"])]
    # Sanity pre kurzy: odstránime nezmysly
    for c in [c for c in df.columns if c.startswith("odds_")]:
        df.loc[(df[c] < 1.001) | (df[c] > 200), c] = np.nan
    df["w_key"] = df["tour"] + "|" + df["winner"].map(td_key)
    df["l_key"] = df["tour"] + "|" + df["loser"].map(td_key)
    counts = pd.concat([df["w_key"], df["l_key"]]).value_counts().to_dict()
    cmap = canonical_map(counts)
    if cmap:
        df["w_key"] = df["w_key"].replace(cmap)
        df["l_key"] = df["l_key"].replace(cmap)
    df = df.sort_values(["date", "tour", "row"], kind="stable").reset_index(drop=True)
    return df
