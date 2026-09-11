"""Povrch a počet víťazných setov podľa turnaja z The Odds API."""

SURFACE_BY_KEY = {
    # ATP
    "tennis_atp_aus_open_singles": "Hard", "tennis_atp_barcelona_open": "Clay", "tennis_atp_canadian_open": "Hard",
    "tennis_atp_china_open": "Hard", "tennis_atp_cincinnati_open": "Hard", "tennis_atp_dubai": "Hard",
    "tennis_atp_french_open": "Clay", "tennis_atp_hamburg_open": "Clay", "tennis_atp_halle_open": "Grass",
    "tennis_atp_indian_wells": "Hard", "tennis_atp_italian_open": "Clay", "tennis_atp_madrid_open": "Clay",
    "tennis_atp_miami_open": "Hard", "tennis_atp_monte_carlo_masters": "Clay", "tennis_atp_munich": "Clay",
    "tennis_atp_paris_masters": "Hard", "tennis_atp_qatar_open": "Hard", "tennis_atp_queens_club_champ": "Grass",
    "tennis_atp_shanghai_masters": "Hard", "tennis_atp_us_open": "Hard", "tennis_atp_washington_open": "Hard",
    "tennis_atp_wimbledon": "Grass",
    # WTA
    "tennis_wta_aus_open_singles": "Hard", "tennis_wta_bad_homburg_open": "Grass", "tennis_wta_canadian_open": "Hard",
    "tennis_wta_charleston_open": "Clay", "tennis_wta_china_open": "Hard", "tennis_wta_cincinnati_open": "Hard",
    "tennis_wta_dubai": "Hard", "tennis_wta_french_open": "Clay", "tennis_wta_german_open": "Grass",
    "tennis_wta_indian_wells": "Hard", "tennis_wta_italian_open": "Clay", "tennis_wta_madrid_open": "Clay",
    "tennis_wta_miami_open": "Hard", "tennis_wta_monterrey_open": "Hard", "tennis_wta_qatar_open": "Hard",
    "tennis_wta_queens_club_champ": "Grass", "tennis_wta_strasbourg": "Clay", "tennis_wta_stuttgart_open": "Clay",
    "tennis_wta_us_open": "Hard", "tennis_wta_wimbledon": "Grass", "tennis_wta_wuhan_open": "Hard",
}

CLAY_WORDS = ["french", "roland", "monte", "madrid", "italian", "rome", "barcelona", "hamburg", "munich", "charleston",
              "strasbourg", "stuttgart", "rio", "buenos", "estoril", "geneva", "lyon", "bastad", "gstaad", "kitzbuhel",
              "umag", "rabat", "bogota", "palermo", "prague", "parma", "bucharest", "clay"]
GRASS_WORDS = ["wimbledon", "halle", "queen", "eastbourne", "hertogenbosch", "mallorca", "berlin", "german_open",
               "bad_homburg", "nottingham", "birmingham", "newport", "grass"]
GRAND_SLAMS = ["aus_open", "french_open", "wimbledon", "us_open"]


def surface_for(sport_key: str, title: str = "") -> str:
    if sport_key in SURFACE_BY_KEY:
        return SURFACE_BY_KEY[sport_key]
    s = (sport_key + " " + title).lower().replace(" ", "_")
    if any(w in s for w in GRASS_WORDS):
        return "Grass"
    if any(w in s for w in CLAY_WORDS):
        return "Clay"
    return "Hard"


def tour_for(sport_key: str) -> str | None:
    if sport_key.startswith("tennis_atp"):
        return "ATP"
    if sport_key.startswith("tennis_wta"):
        return "WTA"
    return None


def best_of_for(sport_key: str) -> int:
    return 5 if sport_key.startswith("tennis_atp") and any(g in sport_key for g in GRAND_SLAMS) else 3
