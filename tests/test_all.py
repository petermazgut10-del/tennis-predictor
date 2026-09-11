"""Testy: python tests/test_all.py  (alebo pytest). Bežia bez internetu, na syntetických dátach."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:  # dá sa spustiť aj bez pytestu
    pytest = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from src.model import bo3_to_bo5  # noqa: E402
from src.names import PlayerIndex, canonical_map, parse_td, td_key  # noqa: E402


# ---------- mená ----------
REAL = {  # tennis-data meno -> meno v The Odds API
    "Alcaraz C.": "Carlos Alcaraz",
    "Auger-Aliassime F.": "Felix Auger-Aliassime",
    "De Minaur A.": "Alex de Minaur",
    "Bautista Agut R.": "Roberto Bautista Agut",
    "Etcheverry T.M.": "Tomas Martin Etcheverry",
    "Zhang Zh.": "Zhizhen Zhang",
    "O'Connell C.": "Christopher O'Connell",
    "Mpetshi Perricard G.": "Giovanni Mpetshi Perricard",
    "Davidovich Fokina A.": "Alejandro Davidovich Fokina",
    "Zverev A.": "Alexander Zverev",
    "Zverev M.": "Mischa Zverev",
    "Pliskova Ka.": "Karolina Pliskova",
    "Pliskova Kr.": "Kristyna Pliskova",
    "Swiatek I.": "Iga Świątek",
    "Wong C.": "Coleman Wong",
}


def test_parse_td():
    assert parse_td("Auger-Aliassime F.") == ("auger aliassime", "f")
    assert parse_td("Etcheverry T.M.") == ("etcheverry", "tm")
    assert td_key("De Minaur A.") == "de minaur|a"


def test_match_real_names():
    players = {}
    for td in REAL:
        tour = "WTA" if td.startswith(("Pliskova", "Swiatek")) else "ATP"
        players[f"{tour}|{td_key(td)}"] = {"name": td, "last": "2026-01-01", "n": 50}
    idx = PlayerIndex(players)
    for td, full in REAL.items():
        tour = "WTA" if td.startswith(("Pliskova", "Swiatek")) else "ATP"
        assert idx.match(full, tour) == f"{tour}|{td_key(td)}", full
    assert idx.match("Unknown Player", "ATP") is None


def test_canonical_merge():
    m = canonical_map({"ATP|etcheverry|t": 5, "ATP|etcheverry|tm": 100, "WTA|pliskova|ka": 50, "WTA|pliskova|kr": 50})
    assert m == {"ATP|etcheverry|t": "ATP|etcheverry|tm"}
    assert canonical_map({"WTA|pliskova|k": 3, "WTA|pliskova|ka": 50, "WTA|pliskova|kr": 50}) == {}


# ---------- matematika ----------
def test_bo5_conversion():
    assert abs(bo3_to_bo5(np.array([0.5]))[0] - 0.5) < 1e-6
    p = bo3_to_bo5(np.array([0.6, 0.7, 0.9]))
    assert (p > np.array([0.6, 0.7, 0.9])).all() and (np.diff(p) > 0).all()
    q = 0.6  # priamo: set 0.6 -> bo3 = q^2(3-2q), bo5 = q^3(10-15q+6q^2)
    assert abs(bo3_to_bo5(np.array([q * q * (3 - 2 * q)]))[0] - q ** 3 * (10 - 15 * q + 6 * q * q)) < 1e-6


# ---------- celá pipeline s falošným The Odds API ----------
class FakeResp:
    def __init__(self, data, remaining=450):
        self.status_code = 200
        self._d = data
        self.text = json.dumps(data)[:200]
        self.headers = {"x-requests-remaining": str(remaining), "x-requests-used": "50"}

    def json(self):
        return self._d


class FakeSession:
    def __init__(self, events, scores=None):
        self.events, self.scores, self.calls = events, scores or [], []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/sports"):
            return FakeResp([{"key": "tennis_atp_china_open", "group": "Tennis", "title": "ATP China Open", "active": True,
                              "has_outrights": False},
                             {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": True, "has_outrights": False}])
        if url.endswith("/odds"):
            return FakeResp(self.events)
        if url.endswith("/scores"):
            return FakeResp(self.scores)
        raise AssertionError(url)


def _event(eid, a, b, oa, ob, start):
    books = [("pinnacle", "Pinnacle", 1.0), ("unibet_eu", "Unibet", 0.97), ("betfair_ex_eu", "Betfair", 1.02)]
    return {"id": eid, "sport_key": "tennis_atp_china_open", "sport_title": "ATP China Open", "commence_time": start,
            "home_team": a, "away_team": b,
            "bookmakers": [{"key": k, "title": t, "last_update": start, "markets": [{"key": "h2h", "outcomes": [
                {"name": a, "price": round(oa * f, 2)}, {"name": b, "price": round(ob * f, 2)}]}]} for k, t, f in books]}


def _workdir():
    d = tempfile.mkdtemp()
    for item in ["config.py", "run.py", "src"]:
        src = os.path.join(ROOT, item)
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, os.path.join(d, item))
    from make_synthetic import generate
    generate(os.path.join(d, "data/raw"), start=2013, seed=3)
    return d


if pytest:
    @pytest.fixture(scope="module")
    def workdir():
        d = _workdir()
        yield d
        shutil.rmtree(d)


def test_pipeline(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    monkeypatch.syspath_prepend(workdir)
    for m in [m for m in sys.modules if m == "config" or m.startswith("src")]:
        del sys.modules[m]
    import config
    import run
    from src import odds_api

    config.ODDS_API_KEY = "test"
    players = pd.read_csv("data/raw/_players_ATP.csv")
    top = players.sort_values("skill", ascending=False)["full"].tolist()
    fut = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [
        _event("e1", top[0], top[40], 3.5, 1.33, fut),      # silný hráč za vysoký kurz -> value
        _event("e2", top[1], top[2], 1.9, 1.9, fut),
        _event("e3", top[3], "Neznamy Hrac", 1.5, 2.6, fut),  # nenapárovaný
        _event("e4", top[5], top[6], 2.0, 1.8, past),       # už začal -> bez value tipu
        {"id": "d1", "home_team": "A B/C D", "away_team": "E F/G H", "commence_time": fut, "bookmakers": []},
    ]
    sess = FakeSession(events)
    monkeypatch.setattr(odds_api.requests, "Session", lambda: sess)
    monkeypatch.setattr(sys, "argv", ["run.py", "update", "--offline"])
    run.main()

    pred = json.load(open("docs/data/predictions.json"))
    ids = {m["id"]: m for m in pred["matches"]}
    assert set(ids) == {"e1", "e2", "e3", "e4"}  # štvorhra vynechaná
    assert ids["e3"]["p"] is None and any("Neznamy" in u for u in pred["unmatched"])
    p1 = ids["e1"]["p"]
    assert abs(sum(p1) - 1) < 1e-9 and p1[0] > 0.6
    assert ids["e1"]["value"] and ids["e1"]["value"]["player"] == top[0]
    assert ids["e4"]["value"] is None
    assert pred["quota_remaining"] == 450
    assert ids["e1"]["odds"][top[0]]["best_book"] == "Betfair"
    bt = json.load(open("docs/data/backtest.json"))
    assert bt["chosen"]["test"]["bets"] > 0
    log = pd.read_csv("state/bet_log.csv")
    assert (log["status"] == "pending").sum() >= 1

    # druhý deň: zápas e1 je odohraný, vyhodnotenie cez /scores
    log.loc[:, "start"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5)).isoformat()
    log.to_csv("state/bet_log.csv", index=False)
    sess.events = []
    sess.scores = [{"id": "e1", "completed": True, "scores": [{"name": top[0], "score": "2"}, {"name": top[40], "score": "1"}]}]
    run.main()
    log = pd.read_csv("state/bet_log.csv")
    r = log[log["bet_id"].str.startswith("e1|")].iloc[0]
    assert r["status"] == "won" and abs(r["profit"] - (r["odds"] - 1)) < 1e-9
    tr = json.load(open("docs/data/tracker.json"))
    assert tr["settled"] >= 1 and tr["won"] >= 1


class _MP:
    """Minimálna náhrada pytest.monkeypatch."""
    def __init__(self):
        self._undo = []

    def chdir(self, d):
        old = os.getcwd()
        os.chdir(d)
        self._undo.append(lambda: os.chdir(old))

    def syspath_prepend(self, d):
        sys.path.insert(0, d)

    def setattr(self, obj, name, val):
        old = getattr(obj, name)
        setattr(obj, name, val)
        self._undo.append(lambda: setattr(obj, name, old))

    def undo(self):
        for f in reversed(self._undo):
            f()


if __name__ == "__main__":
    for t in [test_parse_td, test_match_real_names, test_canonical_merge, test_bo5_conversion]:
        t()
        print("OK", t.__name__)
    d, mp = _workdir(), _MP()
    try:
        test_pipeline(d, mp)
        print("OK test_pipeline")
    finally:
        mp.undo()
        shutil.rmtree(d)
