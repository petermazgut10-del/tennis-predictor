"""Spojenie modelu s trhom ("market-anchored" pravdepodobnosť).

Prečo: backtest na 25 450 zápasoch (2015–2019) ukázal, že samotný model trh neporazí
(presnosť 65,9 % vs 67,8 %, ROI −2 %). Ostré kurzy (Pinnacle) sú lepší odhad než čokoľvek,
čo vieme postaviť zo zadarmo dostupných dát. Preto sa berie férová pravdepodobnosť trhu
ako základ a model ju len jemne opraví:

    logit(p) = w_mkt * logit(p_trh) + w_res * (logit(p_model) - logit(p_trh))

Váhy sa učia z histórie (na tréningových rokoch backtestu). Na reálnych dátach vychádza
w_mkt ≈ 1,03 a w_res ≈ −0,08, čiže "keď je model optimistickejší než trh, mierne uber" –
a je to tak stabilne vo všetkých obdobiach aj na ATP aj na WTA.
"""
from __future__ import annotations

import numpy as np

import config

DEFAULT = (1.0, 0.0)   # čistý trh – bezpečná záloha, keď sa váhy nedajú naučiť


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit(p_model, p_market, seed: int = 3) -> tuple[float, float]:
    """Naučí váhy z histórie. Vstup = pravdepodobnosti z pohľadu SKUTOČNÉHO víťaza."""
    from sklearn.linear_model import LogisticRegression

    p_model = np.asarray(p_model, dtype=float)
    p_market = np.asarray(p_market, dtype=float)
    ok = np.isfinite(p_model) & np.isfinite(p_market)
    p_model, p_market = p_model[ok], p_market[ok]
    if len(p_model) < config.BLEND_MIN_N:
        return DEFAULT
    # polovicu zápasov otočíme, aby model videl aj prehry
    rng = np.random.default_rng(seed)
    flip = rng.random(len(p_model)) < 0.5
    pm = np.where(flip, 1 - p_model, p_model)
    qm = np.where(flip, 1 - p_market, p_market)
    lk, lm = logit(qm), logit(pm)
    X = np.column_stack([lk, lm - lk])
    y = (~flip).astype(int)
    try:
        lr = LogisticRegression(fit_intercept=False, C=1e6, max_iter=2000).fit(X, y)
    except Exception:
        return DEFAULT
    w_mkt = float(np.clip(lr.coef_[0][0], *config.BLEND_W_MKT_RANGE))
    w_res = float(np.clip(lr.coef_[0][1], *config.BLEND_W_RES_RANGE))
    return w_mkt, w_res


def apply(w: tuple[float, float], p_model, p_market):
    """Výsledná pravdepodobnosť. Kde chýba trh, vráti sa model (tipovať sa aj tak nedá)."""
    w_mkt, w_res = w
    pm = np.asarray(p_model, dtype=float)
    qm = np.asarray(p_market, dtype=float)
    out = sigmoid(w_mkt * logit(qm) + w_res * (logit(pm) - logit(qm)))
    return np.where(np.isfinite(qm), out, pm)


def describe(w: tuple[float, float]) -> str:
    if abs(w[1]) < 0.005:
        return "čistý trh (model nepridáva informáciu)"
    smer = "uberá" if w[1] < 0 else "pridáva"
    return f"trh ×{w[0]:.2f}, korekcia modelu {w[1]:+.3f} ({smer} keď sa model líši od trhu)"
