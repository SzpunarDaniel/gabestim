from __future__ import annotations
"""
Car Price Estimator — scraping AutoScout24 (temps réel, pas d'anti-bot)
"""

import requests
from bs4 import BeautifulSoup
import time
import statistics
import json
import re
from dataclasses import dataclass
from typing import Optional
import urllib.parse

# ─────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────

@dataclass
class CarListing:
    title: str
    price: int
    year: Optional[int] = None
    mileage: Optional[int] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    source: str = ""

@dataclass
class CarCriteria:
    brand: str
    model: str
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    mileage_min: Optional[int] = None
    mileage_max: Optional[int] = None
    fuel: Optional[str] = None
    transmission: Optional[str] = None

# ─────────────────────────────────────────────
#  Headers
# ─────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.fr/",
}

# ─────────────────────────────────────────────
#  Mapping marques / modèles → slugs AutoScout24
# ─────────────────────────────────────────────

BRAND_SLUGS = {
    "Alfa Romeo": "alfa-romeo", "Aston Martin": "aston-martin",
    "Audi": "audi", "BMW": "bmw", "Citroën": "citroen", "Cupra": "cupra",
    "Dacia": "dacia", "DS": "ds", "Ferrari": "ferrari", "Fiat": "fiat",
    "Ford": "ford", "Honda": "honda", "Hyundai": "hyundai", "Jaguar": "jaguar",
    "Jeep": "jeep", "Kia": "kia", "Lamborghini": "lamborghini",
    "Land Rover": "land-rover", "Lexus": "lexus", "Maserati": "maserati",
    "Mazda": "mazda", "Mercedes": "mercedes-benz", "Mini": "mini",
    "Mitsubishi": "mitsubishi", "Nissan": "nissan", "Opel": "opel",
    "Peugeot": "peugeot", "Porsche": "porsche", "Renault": "renault",
    "Seat": "seat", "Skoda": "skoda", "Smart": "smart", "Subaru": "subaru",
    "Suzuki": "suzuki", "Tesla": "tesla", "Toyota": "toyota",
    "Volkswagen": "volkswagen", "Volvo": "volvo", "Abarth": "abarth",
}

MODEL_SLUGS = {
    # Renault
    "Clio": "clio", "Mégane": "megane", "Captur": "captur", "Twingo": "twingo",
    "Kadjar": "kadjar", "Scénic": "scenic", "Arkana": "arkana",
    "Austral": "austral", "Talisman": "talisman", "Zoe": "zoe",
    "Mégane E-Tech": "megane-e-tech",
    # Peugeot
    "108": "108", "208": "208", "308": "308", "408": "408", "508": "508",
    "2008": "2008", "3008": "3008", "5008": "5008",
    "107": "107", "e-208": "e-208", "e-2008": "e-2008",
    # Volkswagen
    "Golf": "golf", "Polo": "polo", "Passat": "passat", "Tiguan": "tiguan",
    "T-Roc": "t-roc", "T-Cross": "t-cross", "Touareg": "touareg",
    "Arteon": "arteon", "ID.3": "id.3", "ID.4": "id.4", "ID.5": "id.5", "ID.7": "id.7",
    # BMW
    "Série 1": "1er", "Série 2": "2er", "Série 3": "3er", "Série 4": "4er",
    "Série 5": "5er", "Série 6": "6er", "Série 7": "7er",
    "X1": "x1", "X2": "x2", "X3": "x3", "X4": "x4", "X5": "x5", "X6": "x6",
    "Z4": "z4", "i3": "i3", "i4": "i4", "iX": "ix",
    # Mercedes
    "Classe A": "a-klasse", "Classe B": "b-klasse", "Classe C": "c-klasse",
    "Classe E": "e-klasse", "Classe S": "s-klasse",
    "GLA": "gla", "GLB": "glb", "GLC": "glc", "GLE": "gle", "GLS": "gls",
    "CLA": "cla", "CLS": "cls", "EQA": "eqa", "EQB": "eqb",
    "EQC": "eqc", "EQS": "eqs", "AMG GT": "amg-gt",
    # Audi
    "A1": "a1", "A2": "a2", "A3": "a3", "A4": "a4", "A5": "a5",
    "A6": "a6", "A7": "a7", "A8": "a8",
    "Q2": "q2", "Q3": "q3", "Q4 e-tron": "q4-e-tron",
    "Q5": "q5", "Q7": "q7", "Q8": "q8", "TT": "tt", "R8": "r8",
    # Toyota
    "Yaris": "yaris", "Corolla": "corolla", "C-HR": "c-hr", "RAV4": "rav4",
    "Aygo": "aygo", "Prius": "prius", "Yaris Cross": "yaris-cross", "bZ4X": "bz4x",
    "Land Cruiser": "land-cruiser", "Camry": "camry",
    # Citroën
    "C1": "c1", "C2": "c2", "C3": "c3", "C4": "c4", "C5": "c5",
    "C5 Aircross": "c5-aircross", "C5 X": "c5-x", "C3 Aircross": "c3-aircross",
    "C4 Picasso": "c4-picasso", "Berlingo": "berlingo", "ë-C4": "e-c4",
    # Ford
    "Fiesta": "fiesta", "Focus": "focus", "Puma": "puma", "Kuga": "kuga",
    "Mustang": "mustang", "Mustang Mach-E": "mustang-mach-e",
    "Explorer": "explorer", "Transit": "transit", "EcoSport": "ecosport",
    # Opel
    "Corsa": "corsa", "Astra": "astra", "Insignia": "insignia", "Mokka": "mokka",
    "Crossland": "crossland", "Grandland": "grandland", "Combo": "combo",
    # Dacia
    "Sandero": "sandero", "Logan": "logan", "Duster": "duster",
    "Spring": "spring", "Jogger": "jogger",
    # Kia
    "Picanto": "picanto", "Rio": "rio", "Ceed": "ceed", "Sportage": "sportage",
    "Sorento": "sorento", "Stinger": "stinger", "EV6": "ev6", "Niro": "niro",
    # Hyundai
    "i10": "i10", "i20": "i20", "i30": "i30", "Tucson": "tucson",
    "Santa Fe": "santa-fe", "Kona": "kona", "IONIQ 5": "ioniq-5", "IONIQ 6": "ioniq-6",
    # Fiat
    "500": "500", "Panda": "panda", "Tipo": "tipo", "500X": "500x", "500L": "500l",
    # Seat
    "Ibiza": "ibiza", "Leon": "leon", "Arona": "arona", "Ateca": "ateca", "Tarraco": "tarraco",
    # Skoda
    "Fabia": "fabia", "Octavia": "octavia", "Superb": "superb",
    "Karoq": "karoq", "Kodiaq": "kodiaq", "Enyaq": "enyaq-iv",
    # Mazda
    "Mazda2": "2", "Mazda3": "3", "Mazda6": "6",
    "CX-3": "cx-3", "CX-5": "cx-5", "CX-30": "cx-30", "CX-60": "cx-60", "MX-5": "mx-5",
    # Volvo
    "XC40": "xc40", "XC60": "xc60", "XC90": "xc90",
    "S60": "s60", "S90": "s90", "V60": "v60", "V90": "v90", "C40": "c40", "EX30": "ex30",
    # Tesla
    "Model 3": "model-3", "Model S": "model-s", "Model X": "model-x", "Model Y": "model-y",
    # Mini
    "Mini 3 portes": "mini", "Mini 5 portes": "mini",
    "Countryman": "countryman", "Clubman": "clubman", "Cabrio": "cabrio",
    # Porsche
    "911": "911", "Cayenne": "cayenne", "Macan": "macan",
    "Panamera": "panamera", "Taycan": "taycan", "718": "718",
    # Jeep
    "Renegade": "renegade", "Compass": "compass",
    "Wrangler": "wrangler", "Cherokee": "cherokee", "Grand Cherokee": "grand-cherokee",
    # Honda
    "Civic": "civic", "Jazz": "jazz", "HR-V": "hr-v", "CR-V": "cr-v",
    # Land Rover
    "Defender": "defender", "Discovery": "discovery", "Range Rover": "range-rover",
    "Range Rover Sport": "range-rover-sport", "Evoque": "range-rover-evoque", "Velar": "range-rover-velar",
    # Nissan
    "Micra": "micra", "Juke": "juke", "Qashqai": "qashqai",
    "X-Trail": "x-trail", "Leaf": "leaf", "Ariya": "ariya", "GT-R": "gt-r",
    # Suzuki
    "Swift": "swift", "Vitara": "vitara", "S-Cross": "s-cross", "Jimny": "jimny", "Ignis": "ignis",
    # Mitsubishi
    "Colt": "colt", "Eclipse Cross": "eclipse-cross", "Outlander": "outlander", "ASX": "asx",
    # Subaru
    "Impreza": "impreza", "Legacy": "legacy", "Outback": "outback",
    "Forester": "forester", "XV": "xv", "BRZ": "brz",
    # Lexus
    "IS": "is", "ES": "es", "LS": "ls", "UX": "ux", "NX": "nx", "RX": "rx", "LX": "lx",
    # Jaguar
    "XE": "xe", "XF": "xf", "F-Pace": "f-pace", "E-Pace": "e-pace",
    "I-Pace": "i-pace", "F-Type": "f-type",
    # Maserati
    "Ghibli": "ghibli", "Quattroporte": "quattroporte",
    "Levante": "levante", "GranTurismo": "granturismo",
    # Lamborghini
    "Huracán": "huracan", "Urus": "urus", "Revuelto": "revuelto",
    # Ferrari
    "Roma": "roma", "Portofino": "portofino", "SF90": "sf90", "F8": "f8", "296 GTB": "296-gtb",
    # Alfa Romeo
    "Giulia": "giulia", "Stelvio": "stelvio", "Tonale": "tonale",
    "147": "147", "156": "156", "159": "159", "MiTo": "mito", "Giulietta": "giulietta",
    # Smart
    "Fortwo": "fortwo", "#1": "hash1", "#3": "hash3",
    # Abarth
    "595": "595", "124 Spider": "124-spider",
}

FUEL_MAP = {"essence": "B", "diesel": "D", "hybride": "M", "électrique": "E", "electrique": "E"}
TRANS_MAP = {"manuelle": "M", "automatique": "A"}

def _slug(s: str) -> str:
    return s.lower().replace(" ", "-").replace("_", "-")

# ─────────────────────────────────────────────
#  Construction URL AutoScout24
# ─────────────────────────────────────────────

def build_as24_url(criteria: CarCriteria, page: int = 1) -> str:
    brand_slug = BRAND_SLUGS.get(criteria.brand, _slug(criteria.brand))
    model_slug = MODEL_SLUGS.get(criteria.model, _slug(criteria.model))

    params = {
        "sort": "standard", "desc": "0",
        "ustate": "N,U", "size": "20", "page": page, "cy": "F",
    }
    if criteria.year_min:    params["fregfrom"] = criteria.year_min
    if criteria.year_max:    params["fregto"]   = criteria.year_max
    if criteria.mileage_min: params["kmfrom"]   = criteria.mileage_min
    if criteria.mileage_max: params["kmto"]     = criteria.mileage_max
    if criteria.fuel:
        fc = FUEL_MAP.get(criteria.fuel.lower())
        if fc: params["fuel"] = fc
    if criteria.transmission:
        tc = TRANS_MAP.get(criteria.transmission.lower())
        if tc: params["gear"] = tc

    return f"https://www.autoscout24.fr/lst/{brand_slug}/{model_slug}?" + urllib.parse.urlencode(params)

# ─────────────────────────────────────────────
#  Scraper AutoScout24
# ─────────────────────────────────────────────

def scrape_autoscout24(criteria: CarCriteria, max_pages: int = 4) -> list[CarListing]:
    listings = []
    session  = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, max_pages + 1):
        url = build_as24_url(criteria, page)
        print(f"  🔍 AutoScout24 page {page} — {url}")
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ⚠️  Erreur AutoScout24 : {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        nd   = soup.find("script", id="__NEXT_DATA__")
        if not nd:
            print("  ⚠️  Pas de __NEXT_DATA__")
            break

        try:
            data     = json.loads(nd.string)
            items    = data.get("props", {}).get("pageProps", {}).get("listings", [])
            page_lst = [_parse_as24_item(item, criteria) for item in items]
            # Filtre strict : garde uniquement le bon modèle
            page_lst = [l for l in page_lst if l and _model_matches(l, criteria)]
            listings.extend(page_lst)
            print(f"      → {len(page_lst)} annonces retenues (sur {len(items)})")
            if not items:
                break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠️  Erreur parsing : {e}")
            break

        time.sleep(0.8)

    return listings

def _model_matches(listing: CarListing, criteria: CarCriteria) -> bool:
    """Vérifie que l'annonce correspond bien au modèle recherché."""
    if not listing.title:
        return True
    title_lower = listing.title.lower()
    brand_lower = criteria.brand.lower()
    model_lower = criteria.model.lower()

    # Le titre doit contenir la marque
    if brand_lower not in title_lower:
        return False

    # Correspondance souple du modèle
    # Ex: "Série 3" → cherche "3er", "serie 3", "3 series", "318", "320", "330"...
    model_slug = MODEL_SLUGS.get(criteria.model, "").lower()

    # Cas spéciaux BMW séries
    bmw_series_map = {
        "série 1": ["1er", "serie 1", "116", "118", "120", "125", "130", "m135"],
        "série 2": ["2er", "serie 2", "218", "220", "225", "230", "m235", "m240"],
        "série 3": ["3er", "serie 3", "316", "318", "320", "325", "328", "330", "335", "340", "m3"],
        "série 4": ["4er", "serie 4", "418", "420", "425", "430", "435", "440", "m4"],
        "série 5": ["5er", "serie 5", "518", "520", "523", "525", "528", "530", "535", "540", "550", "m5"],
        "série 6": ["6er", "serie 6", "620", "625", "630", "635", "640", "645", "650", "m6"],
        "série 7": ["7er", "serie 7", "730", "735", "740", "745", "750", "760", "m7"],
    }
    allowed = bmw_series_map.get(model_lower)
    if allowed:
        return any(k in title_lower for k in allowed)

    # Cas général : le slug du modèle ou le nom du modèle doit apparaître dans le titre
    return (model_slug in title_lower or model_lower in title_lower)

def _parse_as24_item(item: dict, criteria: CarCriteria) -> Optional[CarListing]:
    try:
        # ── Prix : "€ 24 990" dans priceFormatted ──
        price_data = item.get("price", {})
        price_str  = price_data.get("priceFormatted", "") if isinstance(price_data, dict) else str(price_data)
        price_str  = re.sub(r"[^\d]", "", price_str)
        if not price_str:
            return None
        price = int(price_str)
        if price < 500 or price > 500_000:
            return None

        # ── Véhicule : make/model/fuel/transmission/mileageInKm ──
        vehicle = item.get("vehicle", {}) or {}
        fuel    = vehicle.get("fuel", "")
        trans   = vehicle.get("transmission", "")

        # Kilométrage : "84 597 km" → 84597
        mileage_raw = vehicle.get("mileageInKm", "") or ""
        mileage_str = re.sub(r"[^\d]", "", mileage_raw)
        mileage     = int(mileage_str) if mileage_str else None

        # ── Année : dans vehicleDetails, ariaLabel="1ère immatriculation", format "03/2022" ──
        year = None
        for detail in (item.get("vehicleDetails") or []):
            if isinstance(detail, dict) and "immatriculation" in detail.get("ariaLabel", "").lower():
                date_str = detail.get("data", "")  # ex: "03/2022"
                parts    = date_str.split("/")
                if len(parts) == 2 and parts[1].isdigit():
                    year = int(parts[1])
                break

        # ── Titre ──
        make    = vehicle.get("make", criteria.brand)
        model_v = vehicle.get("model", criteria.model)
        variant = vehicle.get("modelVersionInput", "") or ""
        title   = f"{make} {model_v} {variant}".strip()

        # ── Location ──
        loc      = item.get("location") or {}
        location = loc.get("city", "") if isinstance(loc, dict) else ""

        # ── URL ──
        url_path = item.get("url", "")
        url      = f"https://www.autoscout24.fr{url_path}" if url_path else None

        return CarListing(
            title=title, price=price, year=year, mileage=mileage,
            fuel=str(fuel), transmission=str(trans),
            location=str(location), url=url, source="AutoScout24",
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return None

# ─────────────────────────────────────────────
#  Alias pour app.py (compatibilité)
# ─────────────────────────────────────────────

def scrape_lbc(criteria: CarCriteria, max_pages: int = 3) -> list[CarListing]:
    """Redirige vers AutoScout24."""
    return scrape_autoscout24(criteria, max_pages=max_pages)

def scrape_lacentrale(criteria: CarCriteria) -> list[CarListing]:
    """Désactivé — AutoScout24 suffit."""
    return []

# ─────────────────────────────────────────────
#  Sauvegarde en base
# ─────────────────────────────────────────────

def save_listings_to_db(listings: list[CarListing], brand: str, model: str) -> int:
    try:
        from database import insert_listings_bulk, mark_outliers, init_db
        init_db()
        rows = [{
            "brand": brand, "model": model, "price": lst.price,
            "year": lst.year, "mileage": lst.mileage, "fuel": lst.fuel,
            "transmission": lst.transmission, "title": lst.title,
            "location": lst.location, "url": lst.url,
            "source": lst.source.lower().replace(" ", "_"),
        } for lst in listings]
        inserted = insert_listings_bulk(rows)
        mark_outliers(brand, model)
        print(f"  💾 {inserted} nouvelles annonces sauvegardées en base.")
        return inserted
    except Exception as e:
        print(f"  ⚠️  Erreur sauvegarde DB : {e}")
        return 0

# ─────────────────────────────────────────────
#  Filtrage & stats
# ─────────────────────────────────────────────

def filter_listings(listings: list[CarListing], criteria: CarCriteria) -> list[CarListing]:
    filtered = []
    for lst in listings:
        if criteria.year_min and lst.year and lst.year < criteria.year_min: continue
        if criteria.year_max and lst.year and lst.year > criteria.year_max: continue
        if criteria.mileage_min and lst.mileage and lst.mileage < criteria.mileage_min: continue
        if criteria.mileage_max and lst.mileage and lst.mileage > criteria.mileage_max: continue
        if criteria.fuel and lst.fuel:
            if criteria.fuel.lower() not in lst.fuel.lower(): continue
        if criteria.transmission and lst.transmission:
            if criteria.transmission.lower() not in lst.transmission.lower(): continue
        filtered.append(lst)
    return filtered

def remove_outliers(prices: list[int]) -> list[int]:
    if len(prices) < 4:
        return prices
    q1 = statistics.quantiles(prices, n=4)[0]
    q3 = statistics.quantiles(prices, n=4)[2]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [p for p in prices if lower <= p <= upper]