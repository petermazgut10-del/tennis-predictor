"""Historické kurzy zo zrkadla dát tennis-data.co.uk na GitHube.

Originálna stránka tennis-data.co.uk je pre servery GitHubu nedostupná (HTTP 503), takže
sa použije verejná kópia tých istých súborov (repozitár 0xsimulacra/MLT, ATP 2001–2019,
WTA 2007–2019). Slúži na backtest: bez historických kurzov sa nedá overiť, či model
dokáže poraziť trh.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

import config

SOURCES = [
    ("ATP", "https://raw.githubusercontent.com/0xsimulacra/MLT/master/df_atp.csv"),
    ("WTA", "https://raw.githubusercontent.com/0xsimulacra/MLT/master/df_wta.csv"),
]
ODDS_PAIRS = {"avg": ("AvgW", "AvgL"), "max": ("MaxW", "MaxL"), "pinnacle": ("PSW", "PSL"), "b365": ("B365W", "B365L")}


def _dir() -> str:
    return os.path.join(config.RAW_DIR, "odds_mirror")


def download(verbose: bool = True) -> None:
    os.makedirs(_dir(), exist_ok=True)
    for tour, url in SOURCES:
        p = os.path.join(_dir(), f"{tour}.csv")
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < 90 * 86400:
            continue
        try:
            r = requests.get(url, timeout=(10, 120))
        except requests.RequestException as e:
            print(f"  ! zrkadlo kurzov {tour}: {e.__class__.__name__}")
            continue
        if r.status_code == 200 and b"Winner" in r.content[:2000]:
            with open(p, "wb") as f:
                f.write(r.content)
            if verbose:
                print(f"  ✓ zrkadlo kurzov {tour} ({len(r.content) // 1024} kB)")
        else:
            print(f"  ! zrkadlo kurzov {tour}: HTTP {r.status_code}")


def load_all() -> pd.DataFrame | None:
    frames = []
    for tour, _ in SOURCES:
        p = os.path.join(_dir(), f"{tour}.csv")
        if not os.path.exists(p):
            continue
        raw = pd.read_csv(p, low_memory=False)
        d = pd.DataFrame(index=raw.index)
        d["tour"] = tour
        d["date"] = pd.to_datetime(raw["Date"], errors="coerce")
        d["winner"] = raw["Winner"].astype(str).str.strip()
        d["loser"] = raw["Loser"].astype(str).str.strip()
        for name, (cw, cl) in ODDS_PAIRS.items():
            d[f"odds_w_{name}"] = pd.to_numeric(raw[cw], errors="coerce") if cw in raw.columns else np.nan
            d[f"odds_l_{name}"] = pd.to_numeric(raw[cl], errors="coerce") if cl in raw.columns else np.nan
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    for c in [c for c in df.columns if c.startswith("odds_")]:
        df.loc[(df[c] < 1.001) | (df[c] > 200), c] = np.nan
    return df[df[[c for c in df.columns if c.startswith("odds_")]].notna().any(axis=1)]


def combine(live: pd.DataFrame | None, mirror: pd.DataFrame | None) -> pd.DataFrame | None:
    """Spojí kurzy zo živého zdroja (ak funguje) a zo zrkadla; duplicity berie zo živého."""
    parts = [x for x in (live, mirror) if x is not None and len(x)]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=["tour", "date", "winner", "loser"], keep="first")
