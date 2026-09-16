"""Nastavenia tenisového predictora. Všetko, čo chceš meniť, je tu."""
import os

# --- Historické dáta ---
# Výsledky: TennisMyLife (denne aktualizované, MIT licencia). Kurzy na backtest: tennis-data.co.uk (voliteľné).
START_YEAR = 2009            # od ktorého roku sa sťahujú výsledky
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
BACKTEST_FIRST_YEAR = 2013   # prvé roky slúžia na "zahriatie" Elo ratingov

# --- Value stávky ---
# STRATÉGIA (od v2): východiskom nie je model, ale TRH. Férová pravdepodobnosť sa vezme
# z ostrých kurzov (Pinnacle / medián kancelárií, bez marže), model ju len jemne opraví
# (váhy sa učia z histórie) a stávka sa robí na NAJLEPŠÍ kurz na trhu. Overené na 25 450
# zápasoch 2015–2019: čistý model −2 % ROI, táto stratégia +3 % ROI.
MARKET_ANCHORED = True
BLEND_MIN_N = 2000           # menej zápasov s kurzami -> váhy sa neučia, použije sa čistý trh
BLEND_W_MKT_RANGE = (0.70, 1.30)
BLEND_W_RES_RANGE = (-0.50, 0.50)

# Na akom kurze sa hľadá value: "max" = najlepší kurz na trhu (pri tejto stratégii nutné),
# "avg" = priemer kancelárií, "pinnacle" = Pinnacle.
VALUE_ODDS_BASIS = "max"
MIN_ODDS = 1.50
MAX_ODDS = 4.00
# Minimálna výhoda (edge = p * kurz - 1). Ak AUTO_TUNE_EDGE=True, prah sa vyberie
# v backteste len na "tréningových" rokoch a overí sa na neskorších rokoch (out-of-sample).
MIN_EDGE = 0.02
AUTO_TUNE_EDGE = True
EDGE_GRID = [0.01, 0.02, 0.03, 0.05, 0.07]
TUNE_LAST_TRAIN_YEAR = 2020  # ladenie prahu na rokoch <= 2020 (ak sú kurzy len staršie, posunie sa automaticky)
# Poistky proti "príliš dobrým" tipom: keď sa kurz rozchádza s ostrým trhom o veľa,
# spravidla trh vie niečo navyše (zranenie, forma) alebo je kurz chyba, ktorá nezostane.
MAX_EDGE = 0.30                 # edge nad týmto je podozrivý, nie výhodný
MAX_MARKET_DISAGREEMENT = 0.15  # max. rozdiel výslednej pravdepodobnosti a férovej pravdepodobnosti trhu
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
