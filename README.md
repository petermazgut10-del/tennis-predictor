# 🎾 Tenisový prediktor

Predikcie tenisových zápasov (ATP + WTA) pred zápasom a hľadanie **value stávok** oproti kurzom stávkových kancelárií.
Beží zadarmo na GitHube: každé ráno sa sám aktualizuje a výsledky ukáže na webovej stránke.

**Čo robí**
1. Stiahne výsledky zápasov od roku 2011 (ATP, Challengery, WTA) z [TennisMyLife](https://stats.tennismylife.org) – zadarmo, MIT licencia, denne aktualizované – a historické kurzy z [tennis-data.co.uk](http://www.tennis-data.co.uk) na backtest (ak je stránka dostupná).
2. Vypočíta **Elo ratingy** každého hráča – celkový a pre každý povrch (hard / antuka / tráva).
3. **Kalibračný model** (logistická regresia) z nich spraví pravdepodobnosť výhry. Model sa vždy učí len na minulosti.
4. **Backtest**: overí, ako by model dopadol v rokoch 2015–dnes, a porovná ho s Pinnacle (najostrejšou kanceláriou).
   Prah výhody sa ladí len na rokoch do 2020, roky 2021+ sú poctivý test.
5. Stiahne kurzy na nadchádzajúce zápasy z [The Odds API](https://the-odds-api.com) (zadarmo 500 kreditov / mesiac)
   a označí zápasy, kde je kurz vyšší ako férový kurz modelu.
6. Každý value tip zapíše ako **papierovú stávku** a po zápase ju vyhodnotí – to je skutočný test bez peňazí.
   Tesne pred začiatkom zápasu uloží aj **záverečný kurz** a spočíta **CLV** (či sa trh po tipe posunul k názoru modelu).
   Každé ráno ukladá kurzy všetkých zápasov → **vlastná história kurzov** a vlastný backtest.
7. Všetko zobrazí na stránke (GitHub Pages) – vrátane **kalkulačky** pre akýkoľvek zápas (aj 250-ky a Challengery).

> The Odds API zadarmo pokrýva len Grand Slamy, turnaje 1000 a 500. Na ostatné zápasy použi kalkulačku a kurzy zo svojej stávkovej kancelárie.

---

## Nastavenie (asi 15 minút, raz)

### 1. API kľúč pre kurzy
1. Choď na https://the-odds-api.com → **Get API Key** → plán **Starter (free)**.
2. Kľúč príde e-mailom. Nikomu ho neposielaj a nedávaj ho do súborov – pôjde do GitHub „Secrets“.

### 2. Repozitár na GitHube
1. Prihlás sa na https://github.com (alebo si vytvor účet).
2. Vpravo hore **+ → New repository**. Názov napr. `tennis-predictor`, nastav **Public** (GitHub Pages zadarmo funguje len pre verejné repozitáre), klikni **Create repository**.
3. Na stránke nového repozitára klikni na odkaz **uploading an existing file**.
4. Rozbaľ ZIP a pretiahni **obsah** priečinka `tennis-predictor` (priečinky `docs`, `src`, `state`, `tests`, súbory `run.py`, `config.py`, …) do okna prehliadača → **Commit changes**.
5. **Priečinok `.github` sa takto často nenahrá** (je skrytý). Preto ho vytvor ručne:
   **Add file → Create new file**, do názvu napíš presne `.github/workflows/daily.yml`,
   skopíruj doň obsah súboru `.github/workflows/daily.yml` zo ZIPu → **Commit changes**.
   *(Na Macu zobrazíš skryté súbory vo Finderi skratkou Cmd + Shift + .)*

### 3. Uloženie API kľúča
**Settings → Secrets and variables → Actions → New repository secret**
- Name: `ODDS_API_KEY`
- Secret: tvoj kľúč z e-mailu → **Add secret**

### 4. Prvé spustenie
1. Záložka **Actions** → ak treba, potvrď „I understand my workflows, go ahead and enable them“.
2. Vľavo **Tenisové predikcie** → **Run workflow** → **Run workflow**.
3. Prvý beh trvá cca 3–6 minút (sťahuje 15 rokov dát). Zelená fajka = hotovo.

### 5. Zapnutie stránky
**Settings → Pages** → Source: **Deploy from a branch** → Branch: `main`, priečinok: **`/docs`** → **Save**.
O minútu-dve bude stránka na `https://<tvoje-meno>.github.io/tennis-predictor/`.

Odteraz sa to každý deň o 7:00 (letný čas) samo aktualizuje. Ručne kedykoľvek cez **Actions → Run workflow**.
Okrem toho beží každé 2 hodiny krátka kontrola: ak niektorý papierový tip začína do ~2 hodín, stiahne aktuálne kurzy (záverečný kurz pre CLV). Keď nič nezačína, nestojí žiadne kredity.

---

## Ako čítať výsledky

| Pojem | Význam |
|---|---|
| **Pravdepodobnosť (model)** | odhad modelu, že hráč vyhrá |
| **Férový kurz** | 1 / pravdepodobnosť – kurz bez marže |
| **Edge (výhoda)** | pravdepodobnosť × kurz − 1. Napr. 55 % × 2,00 − 1 = +10 % |
| **Min. kurz na value** | najnižší kurz, pri ktorom tip ešte spĺňa prah. Porovnaj s kurzom v Niké / Tipos / Fortune |
| **Pinnacle (trh)** | pravdepodobnosť podľa najostrejšej kancelárie – keď sa model s trhom veľmi nezhoduje, častejšie má pravdu trh |
| **Návrh vkladu** | ¼ Kelly, max 2 % bankrollu |

**Verdikt na stránke** (zelený / oranžový pás) hovorí, či backtest ukázal výhodu aj na rokoch, na ktorých sa model neladil.
Aj pri zelenom verdikte platí: **najprv 2–3 mesiace papierových stávok** (aspoň ~100 vyhodnotených tipov). Až keď sú v pluse, má zmysel uvažovať o reálnych peniazoch alebo o platenom API.

---

## Úpravy
Všetky nastavenia sú v `config.py` (dá sa editovať priamo na GitHube – ikona ceruzky):
- `VALUE_ODDS_BASIS` – na akom kurze sa meria value: `"avg"` (priemer, realistické), `"max"` (najlepší kurz), `"pinnacle"`
- `MIN_ODDS`, `MAX_ODDS` – rozsah kurzov
- `AUTO_TUNE_EDGE` / `MIN_EDGE` – automatický alebo pevný prah výhody
- `KELLY_FRACTION`, `MAX_STAKE_PCT` – veľkosť vkladu

**Nenapárovaný hráč** (meno z The Odds API sa nenašlo v historických dátach): doplň riadok do `state/aliases.csv`, napr.
```
tour,odds_name,data_name
ATP,Alex Michelsen,Alex Michelsen
```
(`data_name` = meno tak, ako je v Elo rebríčku na stránke)

**Kredity The Odds API**: jeden beh stojí ~1 kredit za každý práve hraný veľký turnaj + 2 kredity za vyhodnotenie. Pri 1 behu denne to je zvyčajne 100–350 kreditov mesačne. Keď ostane menej ako 40, sťahovanie sa samo preskočí.

## Spustenie na vlastnom počítači (voliteľné)
```
pip install -r requirements.txt
python run.py update            # bez ODDS_API_KEY spraví len históriu + backtest
python tests/test_all.py        # testy na syntetických dátach (bez internetu)
```

## Riešenie problémov
- **Workflow skončí chybou pri „git push“** → Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save.
- **Stránka hlási, že dáta nie sú vygenerované** → workflow ešte nebežal alebo zlyhal (pozri Actions → posledný beh → log).
- **GitHub vypne naplánované behy** po 60 dňoch bez aktivity v repozitári – stačí kliknúť *Enable workflow*.

## Obmedzenia (poctivo)
- Model nevie o zraneniach, únave ani motivácii. Stávkové kancelárie áno.
- Ak tennis-data.co.uk nie je dostupné, backtest proti historickým kurzom sa preskočí (stránka to ukáže). Ratingy a tipy fungujú ďalej.
- Backtest ráta s kurzami tesne pred zápasom; reálne kurzy, za ktoré stihneš staviť, môžu byť iné. Kancelárie tiež obmedzujú úspešných hráčov.
- Hraj zodpovedne a len s peniazmi, ktoré si môžeš dovoliť stratiť.
