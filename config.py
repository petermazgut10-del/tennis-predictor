"""Nastavenia tenisového predictora. Všetko, čo chceš meniť, je tu."""
import os

# --- Historické dáta ---
# Výsledky: TennisMyLife (denne aktualizované, MIT licencia). Kurzy na backtest: tennis-data.co.uk (voliteľné).
START_YEAR = 2011            # od ktorého roku sa sťahujú výsledky
ODDS_START_YEAR = 2013       # od ktorého roku sa sťahujú historické kurzy
ODDS_DOWNLOAD_BUDGET_S = 120 # max. čas na sťahovanie kurzov v jednom behu (pomalý server nezdrží celý beh)
TOURS = ["ATP", "WTA"]
INCLUDE_CHALLENGERS = True   # Challengery zlepšujú ratingy hráčov mimo top 100
RAW_DIR = "data/raw"

# --- Elo ---
ELO_START = 1500.0
ELO_K_NUM = 250.0            # K = ELO_K_NUM / (zápasy + ELO_K_OFFSET) ** ELO_K_SHAPE
ELO_K_OFFSET = 5.0
ELO_K_SHAPE = 0.4
SKIP_WALKOVERS = True        # skreče (walkover) sa do Elo nerátajú
RETIRED_WEIGHT = 0.5         # zápas ukončený skrečom počas hry sa ráta polovičnou váhou

MIN_SERVE_MATCHES = 8        # koľko zápasov so štatistikou podania musí mať hráč, aby sa model bodov použil

# --- Kalibračný model (logistická regresia nad Elo) ---
CALIB_TRAIN_YEARS = 4        # na koľkých predchádzajúcich rokoch sa model učí
BACKTEST_FIRST_YEAR = 2015   # prvé roky slúžia na "zahriatie" Elo ratingov

# --- Value stávky ---
# Na akom kurze sa hľadá value: "avg" = priemerný kurz stávkových kancelárií (realistické),
# "max" = najlepší kurz na trhu (optimistické), "pinnacle" = Pinnacle.
VALUE_ODDS_BASIS = "avg"
MIN_ODDS = 1.30
MAX_ODDS = 5.00
# Minimálna výhoda (edge = p_model * kurz - 1). Ak AUTO_TUNE_EDGE=True, prah sa vyberie
# v backteste len na "tréningových" rokoch a overí sa na neskorších rokoch (out-of-sample).
MIN_EDGE = 0.05
AUTO_TUNE_EDGE = True
EDGE_GRID = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
TUNE_LAST_TRAIN_YEAR = 2020  # ladenie prahu na rokoch <= 2020, test na rokoch po ňom
KELLY_FRACTION = 0.25        # zlomkové Kelly pre návrh veľkosti stávky
MAX_STAKE_PCT = 0.02         # max 2 % bankrollu na jednu stávku

# --- The Odds API (zadarmo 500 kreditov / mesiac) ---
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_REGIONS = "eu"      # eu obsahuje aj Pinnacle a Betfair Exchange
ODDS_API_MIN_REMAINING = 40  # keď ostane menej kreditov, sťahovanie sa preskočí
SETTLE_WITH_SCORES = True    # vyhodnocovať papierové stávky cez /scores (2 kredity / turnaj)
# Priebežné snímky kurzov (príkaz "snapshot", beží každé 2 hodiny): kurzy sa stiahnu len vtedy,
# keď niektorý papierový tip začína v najbližších SNAPSHOT_WINDOW_H hodinách -> záverečný kurz a CLV.
SNAPSHOT_WINDOW_H = 2.25

# --- Výstupy ---
DOCS_DATA_DIR = "docs/data"
STATE_DIR = "state"
