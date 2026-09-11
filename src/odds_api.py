"""Klient pre The Odds API (https://the-odds-api.com) so strážením kreditov."""
from __future__ import annotations

import requests

import config

BASE = "https://api.the-odds-api.com/v4"


class QuotaLow(Exception):
    pass


class OddsAPI:
    def __init__(self, key: str | None = None, session: requests.Session | None = None):
        self.key = key if key is not None else config.ODDS_API_KEY
        self.s = session or requests.Session()
        self.remaining: float | None = None
        self.used: float | None = None
        self.log: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, params: dict, cost: int):
        if self.remaining is not None and self.remaining - cost < config.ODDS_API_MIN_REMAINING:
            raise QuotaLow(f"Zostáva len {self.remaining:.0f} kreditov – preskakujem {path}")
        r = self.s.get(f"{BASE}{path}", params={"apiKey": self.key, **params}, timeout=30)
        h = r.headers
        if "x-requests-remaining" in h:
            try:
                self.remaining = float(h["x-requests-remaining"])
                self.used = float(h.get("x-requests-used", 0))
            except ValueError:
                pass
        if r.status_code != 200:
            self.log.append(f"{path}: HTTP {r.status_code} {r.text[:200]}")
            return None
        return r.json()

    def tennis_sports(self) -> list[dict]:
        """Aktívne tenisové turnaje (tento dopyt kredity nestojí)."""
        data = self._get("/sports", {}, cost=0) or []
        return [s for s in data if s.get("key", "").startswith("tennis_") and s.get("active")
                and not s.get("has_outrights", False)]

    def odds(self, sport_key: str) -> list[dict]:
        return self._get(f"/sports/{sport_key}/odds",
                         {"regions": config.ODDS_API_REGIONS, "markets": "h2h", "oddsFormat": "decimal",
                          "dateFormat": "iso"}, cost=1) or []

    def scores(self, sport_key: str, days_from: int = 3) -> list[dict]:
        return self._get(f"/sports/{sport_key}/scores", {"daysFrom": days_from, "dateFormat": "iso"}, cost=2) or []
