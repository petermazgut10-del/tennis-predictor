"""Kalibračný model: logistická regresia nad Elo rozdielmi a pár ďalšími signálmi.

Všetky vstupy sú antisymetrické (rozdiel hráč A mínus hráč B) a model nemá intercept,
takže P(A vyhrá) + P(B vyhrá) = 1 a nezáleží na tom, kto je "domáci".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["d_all", "d_surf", "d_rank", "d_exp", "d_sexp", "d_days", "d_all_bo5", "d_surf_bo5"]
RANK_FILL = 1500.0


def features(a: dict | pd.DataFrame, b_prefix_swap: bool = False, best_of=3) -> pd.DataFrame:
    """a = tabuľka so stĺpcami a_elo, b_elo, a_surf, ... (výstup z elo.run alebo ručne zložený)."""
    s = pd.DataFrame(a) if isinstance(a, dict) else a
    if b_prefix_swap:
        s = s.rename(columns=lambda c: ("b_" + c[2:]) if c.startswith("a_") else ("a_" + c[2:]) if c.startswith("b_") else c)
    bo5 = (np.asarray(best_of) == 5).astype(float) * np.ones(len(s))
    ar = np.log(s["a_rank"].astype(float).fillna(RANK_FILL).clip(lower=1))
    br = np.log(s["b_rank"].astype(float).fillna(RANK_FILL).clip(lower=1))
    X = pd.DataFrame(index=s.index)
    X["d_all"] = (s["a_elo"] - s["b_elo"]) / 400.0
    X["d_surf"] = (s["a_surf"] - s["b_surf"]) / 400.0
    X["d_rank"] = br - ar
    X["d_exp"] = np.log1p(s["a_n"]) - np.log1p(s["b_n"])
    X["d_sexp"] = np.log1p(s["a_sn"]) - np.log1p(s["b_sn"])
    X["d_days"] = np.log1p(s["a_days"]) - np.log1p(s["b_days"])
    X["d_all_bo5"] = X["d_all"] * bo5
    X["d_surf_bo5"] = X["d_surf"] * bo5
    return X[FEATURES]


def bo3_to_bo5(p3: np.ndarray) -> np.ndarray:
    """Prevod pravdepodobnosti výhry na 2 víťazné sety na 3 víťazné sety (cez pravdepodobnosť setu)."""
    p3 = np.clip(np.asarray(p3, dtype=float), 1e-6, 1 - 1e-6)
    lo, hi = np.zeros_like(p3), np.ones_like(p3)
    for _ in range(50):  # bisekcia: p3 = q^2 (3 - 2q)
        q = (lo + hi) / 2
        f = q * q * (3 - 2 * q)
        lo = np.where(f < p3, q, lo)
        hi = np.where(f >= p3, q, hi)
    q = (lo + hi) / 2
    return q ** 3 * (10 - 15 * q + 6 * q * q)


def elo_prob(X: pd.DataFrame, best_of) -> np.ndarray:
    """Čisté Elo (priemer celkového a povrchového ratingu) – porovnávacia základná línia."""
    p = 1.0 / (1.0 + 10 ** (-(X["d_all"].values + X["d_surf"].values) / 2.0))
    bo5 = np.asarray(best_of) == 5
    return np.where(bo5, bo3_to_bo5(p), p)


def oriented(feats: pd.DataFrame, best_of: np.ndarray, seed: int = 7):
    """Náhodne otočí poradie hráčov, aby model videl výhry aj prehry (y = 1 ak vyhral hráč A)."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(feats)) < 0.5
    Xw = features(feats, best_of=best_of)
    Xl = features(feats, b_prefix_swap=True, best_of=best_of)
    X = pd.DataFrame(np.where(flip[:, None], Xl.values, Xw.values), columns=FEATURES, index=feats.index)
    y = (~flip).astype(int)
    return X, y


class Calibrated:
    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(fit_intercept=False, C=C, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.lr.fit(X.values, y)
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(X[FEATURES].values)[:, 1]

    def to_json(self) -> dict:
        return {"features": FEATURES, "coef": [float(c) for c in self.lr.coef_[0]], "rank_fill": RANK_FILL}
