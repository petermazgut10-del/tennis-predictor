"""Walk-forward backtest: model sa vždy učí len na minulosti a tipuje nasledujúci rok.

Porovnáva model s kurzami Pinnacle (najostrejšia stávková kancelária) a simuluje value stávky.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import blend, markov
from src.model import FEATURES, Calibrated, elo_prob, features, oriented

EPS = 1e-9


def no_vig(o1, o2):
    i1, i2 = 1.0 / o1, 1.0 / o2
    return i1 / (i1 + i2)


def walk_forward(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Vráti df (len zápasy s Elo snapshotom) doplnený o p_model_w a p_elo_w pre testovacie roky."""
    d = df.loc[feats.index].copy()
    d["year"] = d["date"].dt.year
    Xw = features(feats, best_of=d["best_of"].values)
    d["p_elo_w"] = elo_prob(Xw, d["best_of"].values)
    d["p_model_w"] = np.nan
    d["p_noserve_w"] = np.nan   # ten istý model bez informácie o podaní (na porovnanie prínosu)
    base_feats = [f for f in FEATURES if f != "d_markov"]
    years = sorted(d["year"].unique())
    for y in [y for y in years if y >= config.BACKTEST_FIRST_YEAR]:
        tr = d["year"].between(y - config.CALIB_TRAIN_YEARS, y - 1)
        te = d["year"] == y
        if tr.sum() < 500:
            continue
        X, yy = oriented(feats[tr], d.loc[tr, "best_of"].values, seed=y)
        d.loc[te, "p_model_w"] = Calibrated().fit(X, yy).proba(Xw[te])
        d.loc[te, "p_noserve_w"] = Calibrated(feats=base_feats).fit(X, yy).proba(Xw[te])
    return d


def fit_latest(df: pd.DataFrame, feats: pd.DataFrame) -> Calibrated:
    d = df.loc[feats.index]
    year = d["date"].dt.year
    last = year.max()
    tr = year >= last - config.CALIB_TRAIN_YEARS + 1
    # ak je aktuálny rok ešte krátky, zober o rok viac
    if tr.sum() < 3000:
        tr = year >= last - config.CALIB_TRAIN_YEARS
    X, yy = oriented(feats[tr], d.loc[tr, "best_of"].values, seed=int(last))
    return Calibrated().fit(X, yy)


def accuracy_table(d: pd.DataFrame) -> list[dict]:
    """Presnosť na zápasoch s kurzami Pinnacle (férové porovnanie s trhom); bez kurzov na všetkých zápasoch."""
    t = d.dropna(subset=["p_model_w", "odds_w_pinnacle", "odds_l_pinnacle"]).copy()
    methods = [("Model (Elo + podanie)", "p_model_w"), ("Model bez podania", "p_noserve_w"), ("Čisté Elo", "p_elo_w")]
    if "p_blend_w" in t.columns and t["p_blend_w"].notna().any():
        methods.insert(0, ("Zmes trh + model", "p_blend_w"))
    if len(t) >= 200:
        t["p_pin_w"] = no_vig(t["odds_w_pinnacle"], t["odds_l_pinnacle"])
        methods.append(("Pinnacle (trh)", "p_pin_w"))
    else:
        t = d.dropna(subset=["p_model_w"]).copy()
    if t.empty:
        return []
    ar = t["w_rank"].fillna(99999)
    br = t["l_rank"].fillna(99999)
    out = []
    for label, col in methods:
        p = t[col].clip(EPS, 1 - EPS)
        out.append({
            "name": label,
            "accuracy": float((p > 0.5).mean()),
            "log_loss": float(-np.log(p).mean()),
            "brier": float(((1 - p) ** 2).mean()),
            "n": int(len(t)),
        })
    out.append({"name": "Vyššie postavený v rebríčku", "accuracy": float((ar < br).mean() + 0.5 * (ar == br).mean()),
                "log_loss": None, "brier": None, "n": int(len(t))})
    return out


def calibration_bins(d: pd.DataFrame, col: str = "p_model_w", bins: int = 10) -> list[dict]:
    t = d.dropna(subset=[col])
    # obe perspektívy, aby sa pokryl celý rozsah 0–1
    p = np.concatenate([t[col].values, 1 - t[col].values])
    y = np.concatenate([np.ones(len(t)), np.zeros(len(t))])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            out.append({"pred": float(p[m].mean()), "actual": float(y[m].mean()), "n": int(m.sum())})
    return out


def validate_scores(d: pd.DataFrame, feats: pd.DataFrame, sample: int = 6000, seed: int = 11) -> dict:
    """Overí model na úrovni bodov proti skutočnému skóre: počet gemov a 2:0 vs 2:1."""
    idx = d.index[(d["p_model_w"].notna()) & (d.get("score_ok", False) == True)]   # noqa: E712
    f = feats.loc[feats.index.intersection(idx)]
    f = f[(f.get("a_mk", pd.Series(0.5, index=f.index)) != 0.5)]
    if len(f) < 200:
        return {"n": 0}
    if len(f) > sample:
        f = f.sample(sample, random_state=seed)
    dd = d.loc[f.index]
    pa, pb = markov.calibrate_vec(f["a_spw"], f["b_spw"], dd["p_model_w"], dd["best_of"])
    rows = []
    for (p_a, p_b, bo, ga, gl, sw, sl) in zip(pa, pb, dd["best_of"], dd["games_w"], dd["games_l"],
                                              dd["sets_w"], dd["sets_l"]):
        m = markov.match_dist(round(float(p_a), 3), round(float(p_b), 3), int(bo))
        need = int(bo) // 2 + 1
        straight = sum(v for (x, y), v in m["sets"].items() if x + y == need)
        tot = ga + gl
        line = m["exp_games"]
        p_over = sum(v for g, v in m["games"].items() if g > line)
        rows.append((m["exp_games"], tot, straight, int(sw + sl == need), p_over, int(tot > line)))
    r = pd.DataFrame(rows, columns=["exp_games", "games", "p_straight", "straight", "p_over", "over"])
    bins = []
    q = pd.qcut(r["p_over"], 5, duplicates="drop")
    for k, g in r.groupby(q, observed=True):
        bins.append({"pred": float(g["p_over"].mean()), "actual": float(g["over"].mean()), "n": int(len(g))})
    return {
        "n": int(len(r)),
        "pred_games": float(r["exp_games"].mean()), "actual_games": float(r["games"].mean()),
        "mae_games": float((r["exp_games"] - r["games"]).abs().mean()),
        "pred_straight": float(r["p_straight"].mean()), "actual_straight": float(r["straight"].mean()),
        "over_bins": bins,
    }


def market_fair_w(d: pd.DataFrame) -> pd.Series:
    """Férová pravdepodobnosť víťaza podľa trhu (bez marže): Pinnacle, inak priemerné kurzy."""
    fair = no_vig(d["odds_w_pinnacle"], d["odds_l_pinnacle"])
    alt = no_vig(d["odds_w_avg"], d["odds_l_avg"])
    return fair.fillna(alt)


def bet_candidates(d: pd.DataFrame, basis: str, pcol: str = "p_model_w") -> pd.DataFrame:
    """Pre každý zápas vyberie stranu s vyššou výhodou (edge) a odfiltruje podozrivé tipy."""
    ow, ol = d[f"odds_w_{basis}"], d[f"odds_l_{basis}"]
    ok = d[pcol].notna() & ow.notna() & ol.notna()
    ok &= ~d["comment"].str.lower().str.contains("retired|walkover|w/o|disq|award", regex=True)
    t = d[ok].copy()
    pw = t[pcol]
    edge_w = pw * t[f"odds_w_{basis}"] - 1
    edge_l = (1 - pw) * t[f"odds_l_{basis}"] - 1
    pick_w = edge_w >= edge_l
    t["edge"] = np.where(pick_w, edge_w, edge_l)
    t["odds"] = np.where(pick_w, t[f"odds_w_{basis}"], t[f"odds_l_{basis}"])
    t["p"] = np.where(pick_w, pw, 1 - pw)
    t["won"] = pick_w.astype(int)
    t["profit"] = np.where(pick_w, t["odds"] - 1, -1.0)
    fair_w = market_fair_w(t)
    t["p_market"] = np.where(pick_w, fair_w, 1 - fair_w)
    t["disagree"] = (t["p"] - t["p_market"]).abs()
    keep = (t["edge"] <= config.MAX_EDGE) & (t["disagree"].isna() | (t["disagree"] <= config.MAX_MARKET_DISAGREEMENT))
    return t[keep]


def select(c: pd.DataFrame, thr: float) -> pd.DataFrame:
    return c[(c["edge"] >= thr) & (c["odds"] >= config.MIN_ODDS) & (c["odds"] <= config.MAX_ODDS)]


def summarize(b: pd.DataFrame) -> dict:
    n = len(b)
    if n == 0:
        return {"bets": 0, "hit_rate": None, "roi": None, "profit": 0.0, "avg_odds": None, "max_dd": 0.0}
    cum = b["profit"].cumsum().values
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    return {
        "bets": int(n),
        "hit_rate": float(b["won"].mean()),
        "roi": float(b["profit"].sum() / n),
        "profit": float(b["profit"].sum()),
        "avg_odds": float(b["odds"].mean()),
        "max_dd": dd,
    }


def kelly_curve(b: pd.DataFrame) -> list[float]:
    bank, curve = 1.0, []
    for p, o, won in zip(b["p"], b["odds"], b["won"]):
        f = config.KELLY_FRACTION * max(p * o - 1, 0) / (o - 1)
        f = min(f, config.MAX_STAKE_PCT)
        bank *= (1 + f * (o - 1)) if won else (1 - f)
        curve.append(bank)
    return curve


def run(df: pd.DataFrame, feats: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    d = walk_forward(df, feats)
    for c in ("score_ok", "games_w", "games_l", "sets_w", "sets_l"):
        if c in df.columns:
            d[c] = df.loc[d.index, c]
    tested = d[d["p_model_w"].notna()].copy()
    basis = config.VALUE_ODDS_BASIS
    tested["p_mkt_w"] = market_fair_w(tested)

    # hranica tréning/test: ak máme kurzy len pre staršie roky, posunie sa do dát (60 % / 40 %)
    with_odds = tested[tested["p_mkt_w"].notna() & tested[f"odds_w_{basis}"].notna()]
    tune_last = config.TUNE_LAST_TRAIN_YEAR
    if len(with_odds) and tune_last >= with_odds["year"].max():
        tune_last = int(np.quantile(with_odds["year"], 0.6))

    # váhy zmesi (trh + korekcia modelu) sa učia LEN na tréningových rokoch
    tr_mask = with_odds["year"] <= tune_last
    w_blend = blend.fit(with_odds.loc[tr_mask, "p_model_w"], with_odds.loc[tr_mask, "p_mkt_w"])
    tested["p_blend_w"] = blend.apply(w_blend, tested["p_model_w"], tested["p_mkt_w"])

    def tune(pcol):
        c = bet_candidates(tested, basis, pcol)
        tr, te = c[c["year"] <= tune_last], c[c["year"] > tune_last]
        g = [{"edge": t, "train": summarize(select(tr, t)), "test": summarize(select(te, t))}
             for t in config.EDGE_GRID]
        if config.AUTO_TUNE_EDGE and len(tr) >= 200:
            eligible = [x for x in g if x["train"]["bets"] >= 200]
            best = max(eligible or g, key=lambda x: x["train"]["roi"] if x["train"]["roi"] is not None else -9)
            t = best["edge"]
        else:
            t = config.MIN_EDGE
        return c, g, t

    primary = "p_blend_w" if config.MARKET_ANCHORED else "p_model_w"
    strategies = []
    for name, pcol, popis in (("Zmes trh + model", "p_blend_w", "férová p. trhu jemne opravená modelom"),
                              ("Len trh (line shopping)", "p_mkt_w", "stávka, keď je najlepší kurz nad férovým kurzom trhu"),
                              ("Čistý model", "p_model_w", "pôvodná stratégia v1")):
        c_s, g_s, t_s = tune(pcol)
        strategies.append({"name": name, "key": pcol, "popis": popis, "edge": t_s,
                           "train": next(x for x in g_s if x["edge"] == t_s)["train"],
                           "test": next(x for x in g_s if x["edge"] == t_s)["test"],
                           "primary": pcol == primary})
        if pcol == primary:
            cands, grid, thr = c_s, g_s, t_s
    chosen = next(g for g in grid if g["edge"] == thr)

    test_c = cands[cands["year"] > tune_last]
    test_bets = select(test_c, thr)
    all_bets = select(cands, thr)
    per_year = []
    for y, g in all_bets.groupby("year"):
        s = summarize(g)
        s["year"] = int(y)
        s["out_of_sample"] = bool(y > tune_last)
        per_year.append(s)

    def seg(frame, by):
        out = []
        for k, g in frame.groupby(by, observed=True):
            s = summarize(g)
            s["segment"] = str(k)
            out.append(s)
        return out

    test_bets = test_bets.copy()
    test_bets = test_bets.assign(odds_bucket=pd.cut(test_bets["odds"], [1, 1.6, 2.2, 3.0, 5.0, 100],
                                                   labels=["1.30–1.60", "1.60–2.20", "2.20–3.00", "3.00–5.00", "5+"]))
    curve = test_bets[["date", "profit"]].copy()
    curve["cum"] = curve["profit"].cumsum()
    curve = curve.groupby(curve["date"].dt.to_period("W").dt.start_time)["cum"].last().reset_index()

    test_roi = chosen["test"]["roi"]
    train_roi = chosen["train"]["roi"]
    if len(cands) < 500:
        verdict = "no_odds"   # nemáme historické kurzy – nedá sa povedať, či model porazí trh
    elif (test_roi is not None and train_roi is not None and test_roi > 0 and train_roi > 0
          and chosen["test"]["bets"] >= 100):
        verdict = "edge"
    else:
        verdict = "no_edge"

    result = {
        "basis": basis,
        "chosen_edge": thr,
        "strategy": primary,
        "strategies": strategies,
        "blend": {"w_market": round(w_blend[0], 4), "w_model": round(w_blend[1], 4),
                  "popis": blend.describe(w_blend)},
        "tune_last_train_year": tune_last,
        "rules": {"max_edge": config.MAX_EDGE, "max_disagreement": config.MAX_MARKET_DISAGREEMENT,
                  "min_odds": config.MIN_ODDS, "max_odds": config.MAX_ODDS},
        "verdict": verdict,
        "chosen": chosen,
        "grid": grid,
        "per_year": per_year,
        "by_tour": seg(test_bets, "tour"),
        "by_surface": seg(test_bets, "surface"),
        "by_odds": seg(test_bets, "odds_bucket"),
        "accuracy_all": accuracy_table(tested),
        "accuracy_test": accuracy_table(tested[tested["year"] > tune_last]),
        "matches_with_market": int(len(with_odds)),
        "calibration": calibration_bins(tested),
        "scores": validate_scores(d, feats),
        "curve": [{"date": str(r.date.date()), "cum": round(float(r.cum), 2)} for r in curve.itertuples()],
        "kelly_final_bankroll": (kelly_curve(test_bets)[-1] if len(test_bets) else 1.0),
        "years": [int(tested["year"].min()), int(tested["year"].max())] if len(tested) else [],
        "matches_tested": int(len(tested)),
        "matches_with_odds": int(len(cands)),
    }
    return result, d
