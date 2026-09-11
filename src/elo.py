"""Elo ratingy pre tenis: celkové + podľa povrchu (Hard / Clay / Grass).

Prístup podľa FiveThirtyEight / Tennis Abstract: K-faktor klesá s počtom odohraných zápasov,
takže noví hráči sa rýchlo "nájdu" a skúsení sa hýbu pomalšie.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

SURFACES = ("Hard", "Clay", "Grass")


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def k_factor(n: int) -> float:
    return config.ELO_K_NUM / (n + config.ELO_K_OFFSET) ** config.ELO_K_SHAPE


@dataclass
class Player:
    elo: float = config.ELO_START
    n: int = 0
    surf: dict = field(default_factory=lambda: {s: config.ELO_START for s in SURFACES})
    surf_n: dict = field(default_factory=lambda: {s: 0 for s in SURFACES})
    last: pd.Timestamp | None = None
    rank: float = float("nan")
    name: str = ""


class EloBook:
    def __init__(self):
        self.p: dict[str, Player] = {}

    def get(self, key: str) -> Player:
        if key not in self.p:
            self.p[key] = Player()
        return self.p[key]

    def snapshot(self, a: Player, b: Player, surface: str, date) -> dict:
        def days(pl):
            if pl.last is None:
                return 365.0
            return float(min(max((date - pl.last).days, 0), 365))

        return {
            "a_elo": a.elo, "b_elo": b.elo,
            "a_surf": a.surf[surface], "b_surf": b.surf[surface],
            "a_n": a.n, "b_n": b.n,
            "a_sn": a.surf_n[surface], "b_sn": b.surf_n[surface],
            "a_days": days(a), "b_days": days(b),
        }

    def update(self, w: Player, l: Player, surface: str, weight: float = 1.0):
        e = expected(w.elo, l.elo)
        kw, kl = k_factor(w.n) * weight, k_factor(l.n) * weight
        w.elo += kw * (1 - e)
        l.elo -= kl * (1 - e)
        es = expected(w.surf[surface], l.surf[surface])
        kws, kls = k_factor(w.surf_n[surface]) * weight, k_factor(l.surf_n[surface]) * weight
        w.surf[surface] += kws * (1 - es)
        l.surf[surface] -= kls * (1 - es)
        w.n += 1
        l.n += 1
        w.surf_n[surface] += 1
        l.surf_n[surface] += 1


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, EloBook]:
    """Prejde všetky zápasy chronologicky. Pre každý zápas uloží ratingy PRED zápasom
    (a = víťaz, b = porazený), potom ratingy aktualizuje. Žiadny únik budúcich informácií."""
    book = EloBook()
    rows = []
    comments = df["comment"].str.lower().values
    for i, r in enumerate(df.itertuples(index=False)):
        c = comments[i]
        if config.SKIP_WALKOVERS and ("walkover" in c or c.startswith("w/o")):
            rows.append(None)
            continue
        w, l = book.get(r.w_key), book.get(r.l_key)
        snap = book.snapshot(w, l, r.surface, r.date)
        snap["a_rank"] = r.w_rank if not math.isnan(r.w_rank) else w.rank
        snap["b_rank"] = r.l_rank if not math.isnan(r.l_rank) else l.rank
        rows.append(snap)
        weight = config.RETIRED_WEIGHT if "retired" in c else 1.0
        book.update(w, l, r.surface, weight)
        w.last = l.last = r.date
        if not math.isnan(r.w_rank):
            w.rank = r.w_rank
        if not math.isnan(r.l_rank):
            l.rank = r.l_rank
        w.name, l.name = r.winner, r.loser
    keep = [i for i, x in enumerate(rows) if x is not None]
    feats = pd.DataFrame([rows[i] for i in keep], index=df.index[keep])
    return feats, book
