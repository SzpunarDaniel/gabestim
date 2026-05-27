from __future__ import annotations
"""
database.py — Gestionnaire SQLite pour AutoEstim
Stocke toutes les annonces scrapées + les imports de datasets externes.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

# Chemin absolu vers le dossier du script (robuste meme sous OneDrive)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE_DIR, "autoestim.db")
os.makedirs(_BASE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Schéma de la base
# ─────────────────────────────────────────────

SCHEMA = """
-- Table principale : toutes les annonces
CREATE TABLE IF NOT EXISTS listings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identification du véhicule
    brand           TEXT NOT NULL,
    model           TEXT NOT NULL,
    year            INTEGER,
    mileage         INTEGER,
    fuel            TEXT,
    transmission    TEXT,
    color           TEXT,
    doors           INTEGER,
    power_hp        INTEGER,       -- Puissance en chevaux

    -- Prix
    price           INTEGER NOT NULL,

    -- Métadonnées de l'annonce
    title           TEXT,
    location        TEXT,
    url             TEXT UNIQUE,   -- évite les doublons
    source          TEXT,          -- 'leboncoin', 'lacentrale', 'kaggle', etc.
    scraped_at      TEXT,          -- date/heure du scraping

    -- Flags qualité
    is_outlier      INTEGER DEFAULT 0,   -- 1 si détecté comme aberrant
    is_used_for_ml  INTEGER DEFAULT 1    -- 1 si inclus dans l'entraînement ML
);

-- Index pour accélérer les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_brand_model  ON listings (brand, model);
CREATE INDEX IF NOT EXISTS idx_year         ON listings (year);
CREATE INDEX IF NOT EXISTS idx_price        ON listings (price);
CREATE INDEX IF NOT EXISTS idx_source       ON listings (source);

-- Table de suivi des entraînements ML
CREATE TABLE IF NOT EXISTS ml_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    model_type      TEXT NOT NULL,   -- 'price_predictor', 'depreciation'
    n_samples       INTEGER,
    mae             REAL,            -- Mean Absolute Error
    rmse            REAL,            -- Root Mean Squared Error
    r2              REAL,            -- Score R²
    notes           TEXT
);

-- Table de stats agrégées par marque/modèle (cache)
CREATE TABLE IF NOT EXISTS model_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    brand           TEXT NOT NULL,
    model           TEXT NOT NULL,
    year            INTEGER,
    avg_price       REAL,
    median_price    REAL,
    count           INTEGER,
    updated_at      TEXT,
    UNIQUE(brand, model, year)
);
"""

# ─────────────────────────────────────────────
#  Connexion
# ─────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Retourne une connexion SQLite avec row_factory activé."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # accès par nom de colonne
    conn.execute("PRAGMA journal_mode=WAL")   # meilleures perfs en écriture
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    """Crée les tables si elles n'existent pas encore."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    print(f"✅ Base de données initialisée : {DB_PATH}")

# ─────────────────────────────────────────────
#  Insertion des annonces
# ─────────────────────────────────────────────

def insert_listing(
    brand: str,
    model: str,
    price: int,
    year: Optional[int] = None,
    mileage: Optional[int] = None,
    fuel: Optional[str] = None,
    transmission: Optional[str] = None,
    color: Optional[str] = None,
    doors: Optional[int] = None,
    power_hp: Optional[int] = None,
    title: Optional[str] = None,
    location: Optional[str] = None,
    url: Optional[str] = None,
    source: str = "unknown",
) -> Optional[int]:
    """
    Insère une annonce. Ignore silencieusement les doublons (même URL).
    Retourne l'ID inséré, ou None si doublon.
    """
    sql = """
        INSERT OR IGNORE INTO listings
            (brand, model, price, year, mileage, fuel, transmission,
             color, doors, power_hp, title, location, url, source, scraped_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (
            brand.strip().title(),
            model.strip(),
            price,
            year, mileage, fuel, transmission,
            color, doors, power_hp,
            title, location, url, source,
            datetime.now().isoformat(timespec="seconds"),
        ))
        return cur.lastrowid if cur.lastrowid else None

def insert_listings_bulk(rows: list[dict]) -> int:
    """
    Insère une liste de dicts en batch. Plus rapide pour les gros imports.
    Retourne le nombre de lignes réellement insérées.
    """
    sql = """
        INSERT OR IGNORE INTO listings
            (brand, model, price, year, mileage, fuel, transmission,
             color, doors, power_hp, title, location, url, source, scraped_at)
        VALUES
            (:brand, :model, :price, :year, :mileage, :fuel, :transmission,
             :color, :doors, :power_hp, :title, :location, :url, :source, :scraped_at)
    """
    now = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        r.setdefault("color", None)
        r.setdefault("doors", None)
        r.setdefault("power_hp", None)
        r.setdefault("title", None)
        r.setdefault("location", None)
        r.setdefault("url", None)
        r.setdefault("scraped_at", now)

    with get_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        conn.executemany(sql, rows)
        after = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        return after - before

# ─────────────────────────────────────────────
#  Requêtes de lecture
# ─────────────────────────────────────────────

def get_listings_for_ml(
    brand: Optional[str] = None,
    model: Optional[str] = None,
    min_count: int = 0,
) -> list[dict]:
    """
    Retourne les annonces propres pour l'entraînement ML.
    Filtres optionnels par marque/modèle.
    """
    conditions = ["is_outlier = 0", "is_used_for_ml = 1", "price > 0", "year IS NOT NULL"]
    params = []

    if brand:
        conditions.append("LOWER(brand) = LOWER(?)")
        params.append(brand)
    if model:
        conditions.append("LOWER(model) = LOWER(?)")
        params.append(model)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT brand, model, year, mileage, fuel, transmission,
               doors, power_hp, price, source
        FROM listings
        WHERE {where}
        ORDER BY scraped_at DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

def get_stats() -> dict:
    """Retourne des statistiques globales sur la base."""
    with get_connection() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        sources  = conn.execute(
            "SELECT source, COUNT(*) as n FROM listings GROUP BY source ORDER BY n DESC"
        ).fetchall()
        brands   = conn.execute(
            "SELECT COUNT(DISTINCT brand) FROM listings"
        ).fetchone()[0]
        models   = conn.execute(
            "SELECT COUNT(DISTINCT brand || model) FROM listings"
        ).fetchone()[0]
        oldest   = conn.execute("SELECT MIN(scraped_at) FROM listings").fetchone()[0]
        newest   = conn.execute("SELECT MAX(scraped_at) FROM listings").fetchone()[0]

    return {
        "total":   total,
        "brands":  brands,
        "models":  models,
        "sources": [{"source": r["source"], "count": r["n"]} for r in sources],
        "oldest":  oldest,
        "newest":  newest,
    }

def get_price_history(brand: str, model: str) -> list[dict]:
    """
    Retourne l'évolution des prix médians par année de fabrication.
    Utilisé pour la courbe de dépréciation.
    """
    sql = """
        SELECT
            year,
            COUNT(*)            AS count,
            AVG(price)          AS avg_price,
            MIN(price)          AS min_price,
            MAX(price)          AS max_price
        FROM listings
        WHERE LOWER(brand) = LOWER(?)
          AND LOWER(model) = LOWER(?)
          AND is_outlier = 0
          AND year IS NOT NULL
          AND price > 0
        GROUP BY year
        HAVING count >= 2
        ORDER BY year DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (brand, model)).fetchall()
        return [dict(r) for r in rows]

def mark_outliers(brand: str, model: str):
    """
    Marque les annonces aberrantes via la méthode IQR pour une marque/modèle.
    À appeler après chaque batch d'insertion.
    """
    sql_prices = """
        SELECT id, price FROM listings
        WHERE LOWER(brand) = LOWER(?) AND LOWER(model) = LOWER(?)
          AND is_outlier = 0
    """
    with get_connection() as conn:
        rows = conn.execute(sql_prices, (brand, model)).fetchall()
        if len(rows) < 4:
            return

        prices = sorted(r["price"] for r in rows)
        n = len(prices)
        q1 = prices[n // 4]
        q3 = prices[(3 * n) // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_ids = [r["id"] for r in rows if r["price"] < lower or r["price"] > upper]
        if outlier_ids:
            placeholders = ",".join("?" * len(outlier_ids))
            conn.execute(
                f"UPDATE listings SET is_outlier = 1 WHERE id IN ({placeholders})",
                outlier_ids,
            )

# ─────────────────────────────────────────────
#  Import depuis un dataset CSV (Kaggle, etc.)
# ─────────────────────────────────────────────

def _clean_price(raw_str):
    """Nettoie une chaine de prix : '$10,300', '22950.0', '6 800 EUR' -> int."""
    s = str(raw_str or "")
    s = s.replace("$", "").replace("EUR", "").replace("euro", "").replace(",", "").replace(" ", "").replace("\xa0", "").replace("\x00", "").strip()
    try:
        val = int(float(s))
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None

def _clean_mileage(raw_str):
    """Nettoie un kilometrage : '51,000 mi.', '10 500 km', '235000' -> int en km."""
    s = str(raw_str or "").replace("\x00", "").strip()
    is_miles = "mi" in s.lower()
    s = "".join(c for c in s if c.isdigit() or c == ".")
    if not s:
        return None
    try:
        val = int(float(s))
        if is_miles:
            val = int(val * 1.60934)
        return val if val >= 0 else None
    except (ValueError, TypeError):
        return None

def _detect_format(headers):
    """Detecte automatiquement le format du CSV selon ses colonnes."""
    h = {c.lower().strip() for c in headers}
    if "mileage_km" in h and "primary_fuel" in h:
        return "autoscout24_full"
    if "gear" in h and "offertype" in h:
        return "autoscout24_germany"
    if "milage" in h and "fuel_type" in h:
        return "used_cars_us"
    return None

KNOWN_FORMATS = {
    "autoscout24_full": (
        {
            "brand":        "make",
            "model":        "model",
            "year":         "production_year",
            "price":        "price",
            "mileage":      "mileage_km",
            "fuel":         "primary_fuel",
            "transmission": "transmission",
            "power_hp":     "power_hp",
            "doors":        "nr_doors",
        },
        "autoscout24",
    ),
    "autoscout24_germany": (
        {
            "brand":        "make",
            "model":        "model",
            "year":         "year",
            "price":        "price",
            "mileage":      "mileage",
            "fuel":         "fuel",
            "transmission": "gear",
            "power_hp":     "hp",
        },
        "autoscout24_de",
    ),
    "used_cars_us": (
        {
            "brand":        "brand",
            "model":        "model",
            "year":         "model_year",
            "price":        "price",
            "mileage":      "milage",
            "fuel":         "fuel_type",
            "transmission": "transmission",
        },
        "used_cars_us",
    ),
}

def import_from_csv(filepath, column_map=None, source_name=None):
    """
    Importe un dataset CSV dans la base.
    Detecte automatiquement le format si column_map n'est pas fourni.
    Gere : caracteres NUL, encodages varies, prix en $ ou EUR, km en miles.
    """
    import csv

    # Lecture robuste : filtre les caracteres NUL qui corrompent le CSV
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read().replace("\x00", "")
    lines = content.splitlines()
    if not lines:
        print("Fichier vide.")
        return 0

    reader = csv.DictReader(lines)
    headers = reader.fieldnames or []

    # Detection automatique du format
    fmt = _detect_format(headers)
    if fmt and column_map is None:
        column_map, detected_source = KNOWN_FORMATS[fmt]
        if source_name is None:
            source_name = detected_source
        print(f"  Format detecte : {fmt} (source = {source_name})")
    elif column_map is None:
        print(f"  Format inconnu. Colonnes trouvees : {headers[:10]}")
        return 0

    if source_name is None:
        source_name = "csv"

    rows = []
    skipped = 0
    for raw in reader:
        try:
            price = _clean_price(raw.get(column_map["price"], ""))
            if price is None or price < 500 or price > 500000:
                skipped += 1
                continue

            brand = str(raw.get(column_map["brand"], "") or "").replace("\x00", "").strip()
            model = str(raw.get(column_map["model"], "") or "").replace("\x00", "").strip()
            if not brand or not model:
                skipped += 1
                continue

            year_raw = str(raw.get(column_map.get("year", "__"), "") or "").replace("\x00", "")
            year = int(year_raw[:4]) if year_raw and year_raw[:4].isdigit() else None
            if year and (year < 1950 or year > 2026):
                year = None

            mileage = _clean_mileage(raw.get(column_map.get("mileage", "__"), ""))

            fuel = str(raw.get(column_map.get("fuel", "__"), "") or "").replace("\x00", "").strip() or None
            transmission = str(raw.get(column_map.get("transmission", "__"), "") or "").replace("\x00", "").strip() or None

            hp_raw = str(raw.get(column_map.get("power_hp", "__"), "") or "").replace("\x00", "").strip()
            power_hp = int(float(hp_raw)) if hp_raw and hp_raw.replace(".", "").isdigit() else None

            doors_raw = str(raw.get(column_map.get("doors", "__"), "") or "").replace("\x00", "").strip()
            doors = int(float(doors_raw)) if doors_raw and doors_raw.replace(".", "").isdigit() else None

            rows.append({
                "brand":        brand.title(),
                "model":        model,
                "price":        price,
                "year":         year,
                "mileage":      mileage,
                "fuel":         fuel,
                "transmission": transmission,
                "power_hp":     power_hp,
                "doors":        doors,
                "title":        None,
                "location":     None,
                "url":          None,
                "source":       source_name,
                "scraped_at":   datetime.now().isoformat(timespec="seconds"),
            })
        except (ValueError, TypeError):
            skipped += 1
            continue

    print(f"  {len(rows)} lignes valides, {skipped} ignorees.")
    inserted = insert_listings_bulk(rows)
    print(f"Import CSV : {inserted} nouvelles entrees inserees dans la base.")
    return inserted

# ---
#  Point d'entree
# ---

if __name__ == "__main__":
    import sys

    init_db()
    stats = get_stats()
    print(f"\nEtat de la base :")
    print(f"   Total annonces : {stats['total']}")
    print(f"   Marques        : {stats['brands']}")
    print(f"   Modeles        : {stats['models']}")
    for s in stats["sources"]:
        print(f"   Source [{s['source']}] : {s['count']} annonces")

    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        print(f"\nImport du fichier : {csv_path}")
        import_from_csv(csv_path)