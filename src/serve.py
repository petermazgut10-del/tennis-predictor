"""Sila podania a returnu každého hráča.

Pre každý zápas vieme, koľko percent bodov hráč vyhral na vlastnom podaní (spw).
To ale závisí aj od súpera, preto sa udržiavajú dve čísla na hráča:
  s = o koľko je jeho podanie lepšie ako priemer,
  r = o koľko viac bodov berie súperovi na jeho podaní ako priemerný hráč.
Očakávané spw hráča A proti B = priemer povrchu + s(A) − r(B).
Aktualizuje sa chronologicky, takže model nikdy nevidí budúcnosť.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

K0 = 2.0          # rýchlosť učenia: k = K0 / (počet zápasov + K_OFF)
K_OFF = 25.0
CAP = 0.14        # maximálna odchýlka ratingu od priemeru
DEFAULT_BASE = {("ATP", "Hard"): 0.645, ("ATP", "Clay"): 0.625, ("ATP", "Grass"): 0.665,
                ("WTA", "Hard"): 0.575, ("WTA", "Clay"): 0.565, ("WTA", "Grass"): 0.595}


@dataclass
class SrvPlayer:
    s: float = 0.0     # podanie
    r: float = 0.0     # return
    n: int = 0         # počet zápasov so štatistikami


class ServeBook:
    def __init__(self, base: dict | None = None):
        self.p: dict[str, SrvPlayer] = {}
        self.base = dict(DEFAULT_BASE)
        if base:
            self.base.update(base)

    def get(self, key: str) -> SrvPlayer:
        if key not in self.p:
            self.p[key] = SrvPlayer()
        return self.p[key]

    def level(self, tour: str, surface: str) -> float:
        return self.base.get((tour, surface), 0.62 if tour == "ATP" else 0.57)

    def expected(self, a: SrvPlayer, b: SrvPlayer, tour: str, surface: str) -> float:
        """Očakávané % bodov vyhratých na podaní hráča A proti hráčovi B."""
        return float(np.clip(self.level(tour, surface) + a.s - b.r, 0.30, 0.92))

    def update(self, a: SrvPlayer, b: SrvPlayer, obs: float, exp: float, svpt: float):
        w = float(np.clip((svpt or 60) / 60.0, 0.5, 1.3))
        err = obs - exp
        ka = K0 / (a.n + K_OFF) * w
        kb = K0 / (b.n + K_OFF) * w
        a.s = float(np.clip(a.s + ka * err, -CAP, CAP))
        b.r = float(np.clip(b.r - kb * err, -CAP, CAP))


def measure_base(df: pd.DataFrame) -> dict:
    """Priemerné % bodov na podaní podľa okruhu a povrchu (z dát, nie z odhadu)."""
    out = {}
    d = df.dropna(subset=["w_spw", "l_spw"])
    for (tour, surface), g in d.groupby(["tour", "surface"]):
        out[(tour, surface)] = float(pd.concat([g["w_spw"], g["l_spw"]]).mean())
    return out


def run(df: pd.DataFrame, base: dict | None = None) -> tuple[pd.DataFrame, ServeBook]:
    """Chronologicky prejde zápasy. Vráti očakávané spw pred zápasom (a = víťaz, b = porazený)."""
    book = ServeBook(base if base is not None else measure_base(df))
    rows = np.full((len(df), 4), np.nan)
    cols = ["tour", "surface", "w_key", "l_key", "w_spw", "l_spw", "w_svpt", "l_svpt"]
    for i, r in enumerate(df[cols].itertuples(index=False)):
        a, b = book.get(r.w_key), book.get(r.l_key)
        ea = book.expected(a, b, r.tour, r.surface)
        eb = book.expected(b, a, r.tour, r.surface)
        rows[i] = (ea, eb, a.n, b.n)
        if not (np.isnan(r.w_spw) or np.isnan(r.l_spw)):
            book.update(a, b, r.w_spw, ea, r.w_svpt)
            book.update(b, a, r.l_spw, eb, r.l_svpt)
            a.n += 1
            b.n += 1
    return pd.DataFrame(rows, columns=["a_spw", "b_spw", "a_ns", "b_ns"], index=df.index), book
