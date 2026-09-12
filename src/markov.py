"""Model na úrovni bodov: z pravdepodobnosti vyhrať bod na podaní vypočíta celý zápas.

bod -> gem -> tajbrejk -> set -> zápas. Okrem šance na výhru dáva aj rozdelenie skóre
(2:0 / 2:1 …) a počtu gemov – z toho sa neskôr dajú počítať hendikepy a over/under.

Predpoklad (štandardný v literatúre): body sú nezávislé a pravdepodobnosť je konštantná.
V realite to celkom neplatí (dôležité body, momentum), ale odchýlka je malá.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4096)
def game_prob(p: float) -> float:
    """Šanca podávajúceho vyhrať gem pri pravdepodobnosti bodu p."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = 1 - p
    deuce = p * p / (p * p + q * q)  # z 40:40 (deuce)
    # 4:0, 4:1, 4:2 + 40:40
    return (p ** 4
            + 4 * p ** 4 * q
            + 10 * p ** 4 * q ** 2
            + 20 * p ** 3 * q ** 3 * deuce)


@lru_cache(maxsize=4096)
def tiebreak_prob(pa: float, pb: float, target: int = 7) -> float:
    """Šanca hráča A vyhrať tajbrejk. A podáva prvý bod, potom sa strieda po dvoch."""
    # server podľa poradia bodu: A, B,B, A,A, B,B, ...
    def server_a(t: int) -> bool:
        return ((t + 1) // 2) % 2 == 0

    probs = {(0, 0): 1.0}
    win = 0.0
    for t in range(0, 200):
        nxt = {}
        for (a, b), pr in probs.items():
            if a >= target and a - b >= 2:
                continue
            if b >= target and b - a >= 2:
                continue
            p = pa if server_a(t) else 1 - pb  # šanca, že bod vyhrá A
            for res, pp in ((1, p), (0, 1 - p)):
                k = (a + res, b + 1 - res)
                nxt[k] = nxt.get(k, 0.0) + pr * pp
        probs = {}
        for (a, b), pr in nxt.items():
            if a >= target and a - b >= 2:
                win += pr
            elif b >= target and b - a >= 2:
                pass
            else:
                probs[(a, b)] = pr
        if not probs:
            break
    return win


@lru_cache(maxsize=200000)
def set_dist(pa: float, pb: float, a_serves_first: bool = True) -> dict[tuple[int, int], float]:
    """Rozdelenie výsledku setu (gemy A, gemy B) vrátane tajbrejku."""
    ga, gb = game_prob(pa), game_prob(pb)     # šanca podávajúceho vyhrať svoj gem
    out: dict[tuple[int, int], float] = {}
    states = {(0, 0): 1.0}
    for _ in range(13):
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), pr in states.items():
            served = a + b
            a_serving = (served % 2 == 0) == a_serves_first
            p_a_wins_game = ga if a_serving else 1 - gb
            for res, pp in ((1, p_a_wins_game), (0, 1 - p_a_wins_game)):
                na, nb = a + res, b + 1 - res
                if (na >= 6 and na - nb >= 2) or (nb >= 6 and nb - na >= 2) or (na == 7 or nb == 7):
                    out[(na, nb)] = out.get((na, nb), 0.0) + pr * pp
                elif na == 6 and nb == 6:
                    # tajbrejk otvára ten, kto podával prvý gem setu (po 12 gemoch je opäť na rade)
                    tb = tiebreak_prob(pa, pb) if a_serves_first else 1 - tiebreak_prob(pb, pa)
                    out[(7, 6)] = out.get((7, 6), 0.0) + pr * pp * tb
                    out[(6, 7)] = out.get((6, 7), 0.0) + pr * pp * (1 - tb)
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + pr * pp
        states = nxt
        if not states:
            break
    return out


def match_dist(pa: float, pb: float, best_of: int = 3) -> dict:
    """Rozdelenie celého zápasu. Vracia p (šanca A), skóre v setoch a počet gemov."""
    need = best_of // 2 + 1
    d_first = set_dist(pa, pb, True)      # set, v ktorom začína podávať A
    d_second = set_dist(pa, pb, False)
    # stav: (sety A, sety B, kto podáva prvý gem setu) -> {celkové gemy: pravdepodobnosť}
    states = {(0, 0, True): {0: 1.0}}
    p_win = 0.0
    sets_dist: dict[tuple[int, int], float] = {}
    games_dist: dict[int, float] = {}
    for _ in range(best_of):
        nxt: dict[tuple, dict[int, float]] = {}
        for (sa, sb, a_first), gdist in states.items():
            d = d_first if a_first else d_second
            for (ga, gb), ps in d.items():
                nsa, nsb = sa + (ga > gb), sb + (gb > ga)
                # kto podáva prvý gem ďalšieho setu: závisí od parity počtu gemov
                nxt_first = a_first if (ga + gb) % 2 == 0 else not a_first
                total_extra = ga + gb
                key = (nsa, nsb, nxt_first)
                bucket = nxt.setdefault(key, {})
                for g, pg in gdist.items():
                    bucket[g + total_extra] = bucket.get(g + total_extra, 0.0) + pg * ps
        states = {}
        for (sa, sb, a_first), gdist in nxt.items():
            if sa == need or sb == need:
                tot = sum(gdist.values())
                sets_dist[(sa, sb)] = sets_dist.get((sa, sb), 0.0) + tot
                for g, pg in gdist.items():
                    games_dist[g] = games_dist.get(g, 0.0) + pg
                if sa == need:
                    p_win += tot
            else:
                states[(sa, sb, a_first)] = gdist
    exp_games = sum(g * p for g, p in games_dist.items())
    return {"p": p_win, "sets": sets_dist, "games": games_dist, "exp_games": exp_games}


def match_prob(pa: float, pb: float, best_of: int = 3) -> float:
    return match_dist(pa, pb, best_of)["p"]


# ---- rýchly výpočet pre desaťtisíce zápasov: mriežka + lineárna interpolácia ----
GRID = np.round(np.arange(0.30, 0.951, 0.01), 4)


def _build_grid(best_of: int) -> np.ndarray:
    t = np.zeros((len(GRID), len(GRID)))
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            t[i, j] = match_prob(float(a), float(b), best_of)
    return t


_GRIDS: dict[int, np.ndarray] = {}


def match_prob_vec(pa, pb, best_of) -> np.ndarray:
    """Vektorový odhad šance hráča A (interpolácia z mriežky, presnosť ~0,001)."""
    pa = np.clip(np.asarray(pa, dtype=float), GRID[0], GRID[-1])
    pb = np.clip(np.asarray(pb, dtype=float), GRID[0], GRID[-1])
    bo = np.asarray(best_of)
    out = np.empty(len(pa))
    for b in np.unique(bo):
        if int(b) not in _GRIDS:
            _GRIDS[int(b)] = _build_grid(int(b))
        t = _GRIDS[int(b)]
        m = bo == b
        x = (pa[m] - GRID[0]) / 0.01
        y = (pb[m] - GRID[0]) / 0.01
        i0 = np.clip(x.astype(int), 0, len(GRID) - 2)
        j0 = np.clip(y.astype(int), 0, len(GRID) - 2)
        dx, dy = x - i0, y - j0
        out[m] = ((1 - dx) * (1 - dy) * t[i0, j0] + dx * (1 - dy) * t[i0 + 1, j0]
                  + (1 - dx) * dy * t[i0, j0 + 1] + dx * dy * t[i0 + 1, j0 + 1])
    return out


def calibrate(pa: float, pb: float, target_p: float, best_of: int = 3) -> tuple[float, float]:
    """Posunie obe pravdepodobnosti bodu tak, aby šanca na výhru zápasu sedela s modelom.
    Zachová rozdiel v „dominancii podania“, ktorý určuje počet gemov."""
    lo, hi = -0.25, 0.25
    for _ in range(40):
        d = (lo + hi) / 2
        p = match_prob(min(max(pa + d, 0.05), 0.98), min(max(pb - d, 0.05), 0.98), best_of)
        if p < target_p:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return min(max(pa + d, 0.05), 0.98), min(max(pb - d, 0.05), 0.98)


def calibrate_vec(pa, pb, target, best_of):
    """Vektorová verzia calibrate(): posunie podanie oboch hráčov tak, aby šanca sedela s modelom."""
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    target = np.asarray(target, dtype=float)
    bo = np.asarray(best_of)
    lo = np.full(len(pa), -0.25)
    hi = np.full(len(pa), 0.25)
    for _ in range(16):
        d = (lo + hi) / 2
        p = match_prob_vec(np.clip(pa + d, 0.05, 0.98), np.clip(pb - d, 0.05, 0.98), bo)
        lo = np.where(p < target, d, lo)
        hi = np.where(p < target, hi, d)
    d = (lo + hi) / 2
    return np.clip(pa + d, 0.05, 0.98), np.clip(pb - d, 0.05, 0.98)
