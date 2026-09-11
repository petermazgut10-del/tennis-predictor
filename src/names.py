"""Párovanie mien hráčov.

tennis-data.co.uk píše mená ako "Alcaraz C.", "Auger-Aliassime F.", "Etcheverry T.M.".
The Odds API ich píše celé: "Carlos Alcaraz", "Felix Auger-Aliassime", "Tomas Martin Etcheverry".
"""
from __future__ import annotations

import csv
import os
import re
import unicodedata
from collections import defaultdict


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ").replace("'", "").replace("`", "").replace("’", "")
    s = re.sub(r"[^a-z. ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_td(name: str) -> tuple[str, str]:
    """'Auger-Aliassime F.' -> ('auger aliassime', 'f'); 'Etcheverry T.M.' -> ('etcheverry', 'tm')."""
    toks = norm(name).split()
    if not toks:
        return "", ""
    if len(toks) >= 2 and ("." in toks[-1] or len(toks[-1]) <= 2):
        init = toks[-1].replace(".", "")
        surname = " ".join(t.replace(".", "") for t in toks[:-1])
    else:
        init, surname = "", " ".join(t.replace(".", "") for t in toks)
    return surname, init


def td_key(name: str) -> str:
    surname, init = parse_td(name)
    return f"{surname}|{init}"


def _compat(first_tokens: list[str], init: str) -> int:
    """2 = silná zhoda krstného mena s iniciálami, 1 = zhoduje sa prvé písmeno, 0 = nie."""
    if not first_tokens:
        return 0
    if not init:
        return 1
    initials = "".join(t[0] for t in first_tokens)
    joined = "".join(first_tokens)
    if initials == init or initials.startswith(init) or joined.startswith(init) or first_tokens[0].startswith(init):
        return 2
    if initials[0] == init[0]:
        return 1
    return 0


def canonical_map(keys_with_counts: dict[str, int]) -> dict[str, str]:
    """Zlúči varianty toho istého hráča v rámci jedného okruhu, napr. 'etcheverry|t' -> 'etcheverry|tm'.
    Zlučuje sa len keď je to jednoznačné (jediný dlhší variant s rovnakou predponou)."""
    by_surname = defaultdict(list)
    for k in keys_with_counts:
        tour, rest = k.split("|", 1)
        surname, init = rest.split("|")
        by_surname[(tour, surname)].append(init)
    mapping = {}
    for (tour, surname), inits in by_surname.items():
        if len(inits) < 2:
            continue
        for a in inits:
            longer = [b for b in inits if b != a and b.startswith(a)]
            if len(longer) == 1 and a:
                b = longer[0]
                # iba ak nie je iný variant, ktorý by tiež mohol byť "a"
                mapping[f"{tour}|{surname}|{a}"] = f"{tour}|{surname}|{b}"
    return mapping


class PlayerIndex:
    """Index hráčov z histórie pre párovanie celých mien z The Odds API."""

    def __init__(self, players: dict[str, dict], aliases_path: str | None = None):
        # players: key -> {"name": "Alcaraz C.", "last": Timestamp, "n": int}
        self.players = players
        self.by_surname = defaultdict(list)
        for key, info in players.items():
            tour, surname, init = key.split("|")
            self.by_surname[(tour, surname)].append((key, init, info))
        self.aliases = {}
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.aliases[(row["tour"].strip().upper(), norm(row["odds_name"]))] = (
                        f"{row['tour'].strip().upper()}|{td_key(row['td_name'])}"
                    )

    def match(self, full_name: str, tour: str) -> str | None:
        n = norm(full_name).replace(".", "")
        if (tour, n) in self.aliases:
            return self.aliases[(tour, n)]
        toks = n.split()
        if len(toks) < 2:
            return None
        cands = []
        splits = [(toks[:k], toks[k:]) for k in range(1, len(toks))]          # "Carlos | Alcaraz"
        splits += [(toks[k:], toks[:k]) for k in range(1, len(toks))]         # "Zhang | Zhizhen" (priezvisko prvé)
        for order, (first, last) in enumerate(splits):
            surname = " ".join(last)
            for key, init, info in self.by_surname.get((tour, surname), []):
                score = _compat(first, init)
                if score:
                    # silná zhoda > slabá; prirodzené poradie mena > obrátené; viac zápasov a novší hráč vyhráva
                    cands.append((score, order < len(toks) - 1, str(info.get("last", "")), info.get("n", 0), key))
        if not cands:
            return None
        cands.sort(reverse=True)
        best = cands[0]
        # ak sú dve rovnako dobré silné zhody s rôznymi hráčmi, je to nejednoznačné
        ties = [c for c in cands if c[:2] == best[:2] and c[4] != best[4]]
        if ties and best[0] == 1:
            return None
        return best[4]


class FullNameIndex:
    """Index hráčov s celými menami (TennisMyLife) – párovanie mien z The Odds API aj z tennis-data.co.uk."""

    def __init__(self, players: dict[str, dict], aliases_path: str | None = None):
        # players: key ("ATP|S0AG") -> {"name": "Jannik Sinner", "last": Timestamp, "n": int}
        self.players = players
        self.exact = defaultdict(list)
        self.tokset = defaultdict(list)
        self.by_last = defaultdict(list)
        for key, info in players.items():
            tour = key.split("|", 1)[0]
            toks = norm(info.get("name", "")).replace(".", "").split()
            if not toks:
                continue
            rec = (key, toks, info)
            self.exact[(tour, " ".join(toks))].append(rec)
            self.tokset[(tour, " ".join(sorted(toks)))].append(rec)
            self.by_last[(tour, toks[-1])].append(rec)
            if len(toks) >= 3:  # viacslovné priezvisko, napr. "de minaur", "bautista agut"
                self.by_last[(tour, toks[1])].append(rec)
        self.aliases = {}
        if aliases_path and os.path.exists(aliases_path):
            with open(aliases_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    t = row["tour"].strip().upper()
                    target = norm(row.get("data_name") or row.get("td_name") or "").replace(".", "")
                    hit = self.exact.get((t, target))
                    if hit:
                        self.aliases[(t, norm(row["odds_name"]).replace(".", ""))] = self._best(hit)

    @staticmethod
    def _best(recs):
        return max(recs, key=lambda r: (str(r[2].get("last", "")), r[2].get("n", 0)))[0]

    def match(self, full_name: str, tour: str) -> str | None:
        n = norm(full_name).replace(".", "")
        if (tour, n) in self.aliases:
            return self.aliases[(tour, n)]
        toks = n.split()
        if not toks:
            return None
        for table, k in ((self.exact, " ".join(toks)), (self.tokset, " ".join(sorted(toks)))):
            hit = table.get((tour, k))
            if hit:
                return self._best(hit)
        # záloha: rovnaké priezvisko + zhodné krstné meno / iniciála
        cands = []
        for k in range(1, len(toks)):
            first, last = toks[:k], toks[k:]
            for key, ptoks, info in self._by_surname(tour, last):
                pfirst = ptoks[: len(ptoks) - len(last)]
                score = 2 if pfirst and first[0] == pfirst[0] else (1 if pfirst and first[0][0] == pfirst[0][0] else 0)
                if score:
                    cands.append((score, str(info.get("last", "")), info.get("n", 0), key))
        if not cands:
            return None
        cands.sort(reverse=True)
        if len(cands) > 1 and cands[0][0] == 1 and cands[1][0] == 1 and cands[0][3] != cands[1][3]:
            return None  # nejednoznačné
        return cands[0][3]

    def _by_surname(self, tour, last_toks):
        seen = set()
        for rec in self.by_last.get((tour, last_toks[-1]), []) + self.by_last.get((tour, last_toks[0]), []):
            key, ptoks, info = rec
            if key in seen or len(ptoks) <= len(last_toks):
                continue
            if ptoks[-len(last_toks):] == last_toks:
                seen.add(key)
                yield rec

    def match_td(self, td_name: str, tour: str, near=None) -> str | None:
        """'Auger-Aliassime F.' -> kľúč hráča. near = dátum zápasu (preferuje hráča aktívneho v tom čase)."""
        surname, init = parse_td(td_name)
        if not surname:
            return None
        last = surname.split()
        cands = []
        for key, ptoks, info in self._by_surname(tour, last):
            first = ptoks[: len(ptoks) - len(last)]
            score = _compat(first, init)
            if score:
                cands.append((score, info.get("n", 0), key))
        if not cands:
            return None
        cands.sort(reverse=True)
        if len(cands) > 1 and cands[0][0] == cands[1][0] and cands[0][2] != cands[1][2]:
            return None
        return cands[0][2]
