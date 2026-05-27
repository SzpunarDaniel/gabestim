from __future__ import annotations
"""Flask server - Car Price Estimator (AutoScout24 + SQLite + ML)"""

from flask import Flask, render_template, request, jsonify
import sys, os, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from car_price_estimator import (
    CarCriteria, scrape_autoscout24,
    filter_listings, remove_outliers, save_listings_to_db
)
from ml_pipeline import (
    predict_price, predict_depreciation,
    is_model_trained, get_model_meta, train_all
)

app = Flask(__name__)

CAR_DATA = {
    "Abarth": ["500", "595", "124 Spider"],
    "Alfa Romeo": ["Giulia", "Stelvio", "Tonale", "147", "156", "159", "MiTo", "Giulietta"],
    "Audi": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q4 e-tron", "Q5", "Q7", "Q8", "TT", "R8", "e-tron"],
    "BMW": ["Série 1", "Série 2", "Série 3", "Série 4", "Série 5", "Série 6", "Série 7", "X1", "X2", "X3", "X4", "X5", "X6", "Z4", "i3", "i4", "iX"],
    "Citroën": ["C1", "C2", "C3", "C3 Aircross", "C4", "C4 Picasso", "C5", "C5 Aircross", "C5 X", "C6", "Berlingo", "ë-C4"],
    "Dacia": ["Sandero", "Logan", "Duster", "Spring", "Jogger", "Bigster"],
    "Ferrari": ["Roma", "Portofino", "SF90", "F8", "296 GTB"],
    "Fiat": ["500", "500X", "500L", "Panda", "Tipo", "Bravo", "Punto"],
    "Ford": ["Fiesta", "Focus", "Puma", "Kuga", "Mustang", "Mustang Mach-E", "Explorer", "Transit", "EcoSport"],
    "Honda": ["Civic", "Jazz", "HR-V", "CR-V", "e"],
    "Hyundai": ["i10", "i20", "i30", "Tucson", "Santa Fe", "Kona", "IONIQ 5", "IONIQ 6"],
    "Jaguar": ["XE", "XF", "F-Pace", "E-Pace", "I-Pace", "F-Type"],
    "Jeep": ["Renegade", "Compass", "Wrangler", "Cherokee", "Grand Cherokee"],
    "Kia": ["Picanto", "Rio", "Ceed", "Sportage", "Sorento", "Stinger", "EV6", "Niro"],
    "Lamborghini": ["Huracán", "Urus", "Revuelto"],
    "Land Rover": ["Defender", "Discovery", "Range Rover", "Range Rover Sport", "Evoque", "Velar"],
    "Lexus": ["IS", "ES", "LS", "UX", "NX", "RX", "LX"],
    "Maserati": ["Ghibli", "Quattroporte", "Levante", "GranTurismo"],
    "Mazda": ["Mazda2", "Mazda3", "Mazda6", "CX-3", "CX-5", "CX-30", "CX-60", "MX-5"],
    "Mercedes": ["Classe A", "Classe B", "Classe C", "Classe E", "Classe S", "CLA", "CLS", "GLA", "GLB", "GLC", "GLE", "GLS", "AMG GT", "EQA", "EQB", "EQC", "EQS"],
    "Mini": ["Mini 3 portes", "Mini 5 portes", "Cabrio", "Clubman", "Countryman"],
    "Mitsubishi": ["Colt", "Eclipse Cross", "Outlander", "ASX", "L200"],
    "Nissan": ["Micra", "Juke", "Qashqai", "X-Trail", "Leaf", "Ariya", "GT-R"],
    "Opel": ["Corsa", "Astra", "Insignia", "Mokka", "Crossland", "Grandland", "Combo"],
    "Peugeot": ["107", "108", "208", "308", "408", "508", "2008", "3008", "5008", "e-208", "e-2008"],
    "Porsche": ["911", "Cayenne", "Macan", "Panamera", "Taycan", "718"],
    "Renault": ["Twingo", "Clio", "Captur", "Mégane", "Arkana", "Austral", "Kadjar", "Scénic", "Talisman", "Zoe", "Mégane E-Tech"],
    "Seat": ["Ibiza", "Leon", "Arona", "Ateca", "Tarraco"],
    "Skoda": ["Fabia", "Octavia", "Superb", "Karoq", "Kodiaq", "Enyaq"],
    "Smart": ["Fortwo", "#1", "#3"],
    "Subaru": ["Impreza", "Legacy", "Outback", "Forester", "XV", "BRZ"],
    "Suzuki": ["Swift", "Vitara", "S-Cross", "Jimny", "Ignis"],
    "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
    "Toyota": ["Aygo", "Yaris", "Yaris Cross", "Corolla", "C-HR", "RAV4", "Land Cruiser", "Camry", "GR86", "bZ4X", "Prius"],
    "Volkswagen": ["Polo", "Golf", "Passat", "Arteon", "T-Roc", "T-Cross", "Tiguan", "Touareg", "ID.3", "ID.4", "ID.5", "ID.7"],
    "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90", "C40", "EX30", "EX90"],
}

FUELS         = ["Tous", "Essence", "Diesel", "Hybride", "Électrique"]
TRANSMISSIONS = ["Toutes", "Manuelle", "Automatique"]
CURRENT_YEAR  = 2025

@app.route("/")
def index():
    return render_template("index.html",
        brands=sorted(CAR_DATA.keys()),
        fuels=FUELS, transmissions=TRANSMISSIONS,
        years=list(range(CURRENT_YEAR, 1989, -1)))

@app.route("/api/models/<brand>")
def get_models(brand):
    return jsonify(sorted(CAR_DATA.get(brand, [])))

@app.route("/api/estimate", methods=["POST"])
def estimate():
    data = request.json

    brand        = data.get("brand", "").strip()
    model        = data.get("model", "").strip()
    year_min     = int(data["year_min"])    if data.get("year_min")    else None
    year_max     = int(data["year_max"])    if data.get("year_max")    else None
    mileage_min  = int(data["mileage_min"]) if data.get("mileage_min") else None
    mileage_max  = int(data["mileage_max"]) if data.get("mileage_max") else None
    fuel_raw     = data.get("fuel", "Tous")
    trans_raw    = data.get("transmission", "Toutes")
    fuel         = None if fuel_raw  in ("Tous", "")   else fuel_raw.lower()
    transmission = None if trans_raw in ("Toutes", "") else trans_raw.lower()

    if not brand or not model:
        return jsonify({"error": "Marque et modèle requis."}), 400

    criteria = CarCriteria(
        brand=brand, model=model,
        year_min=year_min, year_max=year_max,
        mileage_min=mileage_min, mileage_max=mileage_max,
        fuel=fuel, transmission=transmission,
    )

    # ── Scraping AutoScout24 ──
    all_listings = scrape_autoscout24(criteria, max_pages=4)

    # Sauvegarde en base pour enrichir le ML
    if all_listings:
        save_listings_to_db(all_listings, brand, model)

    filtered = filter_listings(all_listings, criteria)

    if not filtered:
        return jsonify({"error": "Aucune annonce trouvée sur AutoScout24. Essayez d'élargir vos critères."}), 404

    prices_raw   = [l.price for l in filtered if l.price]
    prices_clean = remove_outliers(prices_raw)

    if len(prices_clean) < 2:
        return jsonify({"error": "Pas assez de données pour calculer une estimation fiable."}), 404

    mean_price   = int(statistics.mean(prices_clean))
    median_price = int(statistics.median(prices_clean))
    min_price    = min(prices_clean)
    max_price    = max(prices_clean)
    stdev        = int(statistics.stdev(prices_clean))
    low          = max(0, int(median_price - stdev * 0.5))
    high         = int(median_price + stdev * 0.5)

    listings_sorted = sorted(filtered, key=lambda l: abs(l.price - median_price))
    top_listings = [{
        "title": lst.title, "price": lst.price, "year": lst.year,
        "mileage": lst.mileage, "fuel": lst.fuel,
        "transmission": lst.transmission, "location": lst.location,
        "url": lst.url, "source": lst.source,
    } for lst in listings_sorted[:8]]

    # ── Prédictions ML ──
    ml_prediction   = None
    ml_depreciation = []

    # On prend l'année et le km médians des annonces trouvées
    years_found   = [l.year    for l in filtered if l.year]
    mileages_found= [l.mileage for l in filtered if l.mileage]
    ref_year      = int(statistics.median(years_found))    if years_found    else (year_min or year_max or 2018)
    ref_mileage   = int(statistics.median(mileages_found)) if mileages_found else (mileage_max or mileage_min or 80_000)

    # ── Prédiction IA : régression linéaire sur les annonces scrapées ──
    # Plus fiable que le modèle global car ancré sur les données réelles actuelles
    try:
        ml_prediction, ml_depreciation = _predict_from_listings(
            filtered, ref_year, ref_mileage, median_price
        )
    except Exception as e:
        print(f"  ⚠️  Erreur prédiction locale : {e}")
        ml_prediction   = None
        ml_depreciation = _simple_depreciation(median_price, ref_year)

    # Données pour le scatter plot (année + prix de chaque annonce)
    scatter_data = [
        {"year": l.year, "price": l.price, "title": l.title,
         "mileage": l.mileage, "fuel": l.fuel}
        for l in filtered if l.year and l.price
    ]

    return jsonify({
        "total_raw":       len(all_listings),
        "total_filtered":  len(filtered),
        "total_used":      len(prices_clean),
        "scatter_data":    scatter_data,
        "mean":            mean_price,
        "median":          median_price,
        "min":             min_price,
        "max":             max_price,
        "stdev":           stdev,
        "range_low":       low,
        "range_high":      high,
        "listings":        top_listings,
        "histogram":       _build_histogram(prices_clean, bins=8),
        "ml_prediction":   ml_prediction,
        "ml_depreciation": ml_depreciation,
        "ml_available":    is_model_trained(),
    })

def _build_histogram(prices: list, bins: int = 8) -> list:
    if not prices: return []
    mn, mx = min(prices), max(prices)
    if mn == mx: return [{"label": f"{mn:,} €", "count": len(prices)}]
    step = (mx - mn) / bins
    buckets = []
    for i in range(bins):
        lo    = int(mn + i * step)
        count = sum(1 for p in prices if lo <= p < int(mn + (i+1) * step))
        if i == bins - 1: count += sum(1 for p in prices if p == mx)
        buckets.append({"label": f"{lo // 1000}k" if lo >= 1000 else str(lo), "count": count})
    return buckets

@app.route("/api/db-stats")
def db_stats():
    from database import get_stats
    return jsonify(get_stats())

@app.route("/api/ml-status")
def ml_status():
    return jsonify({"trained": is_model_trained(), "runs": get_model_meta()})

@app.route("/api/train", methods=["POST"])
def train():
    import threading
    threading.Thread(target=lambda: train_all(), daemon=True).start()
    return jsonify({"status": "started"})



def _predict_from_listings(listings, ref_year, ref_mileage, median_price):
    """
    Prédit le prix à partir des annonces scrapées via régression linéaire simple.
    Beaucoup plus précis que le modèle global car ancré sur les données actuelles.
    """
    import datetime
    cur_year = datetime.datetime.now().year

    # Données valides : année + kilométrage + prix
    valid = [(l.year, l.mileage or 80000, l.price)
             for l in listings if l.year and l.price and l.mileage]

    predicted_price = median_price  # fallback

    if len(valid) >= 5:
        import numpy as np
        years    = np.array([v[0] for v in valid], dtype=float)
        mileages = np.array([v[1] for v in valid], dtype=float)
        prices   = np.array([v[2] for v in valid], dtype=float)

        # Régression linéaire multiple : prix ~ année + km
        X = np.column_stack([
            np.ones(len(valid)),
            years - years.mean(),
            mileages / 10000,
        ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, prices, rcond=None)
            age_effect = (ref_year - years.mean()) * coeffs[1]
            km_effect  = (ref_mileage / 10000) * coeffs[2]
            predicted_price = int(coeffs[0] + age_effect + km_effect)
            predicted_price = max(500, min(predicted_price, 500_000))
        except Exception:
            predicted_price = median_price

    margin = int(predicted_price * 0.12)
    ml_prediction = {
        "predicted_price": predicted_price,
        "confidence_low":  max(0, predicted_price - margin),
        "confidence_high": predicted_price + margin,
        "age": cur_year - ref_year,
    }

    # Courbe de dépréciation basée sur le prix prédit
    ml_depreciation = _simple_depreciation(predicted_price, ref_year)
    return ml_prediction, ml_depreciation


def _simple_depreciation(base_price, ref_year):
    """Courbe de dépréciation empirique."""
    import datetime
    cur_year = datetime.datetime.now().year
    cur_age  = cur_year - ref_year
    result   = []
    current  = base_price
    for delta in range(11):
        age  = cur_age + delta
        if delta > 0:
            rate = 0.15 if age <= 3 else (0.10 if age <= 6 else 0.07)
            current = int(current * (1 - rate))
        pct = round(max(500, current) / base_price * 100, 1) if base_price > 0 else 0
        result.append({
            "calendar_year":   cur_year + delta,
            "age":             age,
            "mileage_est":     15000 * (cur_age + delta),
            "predicted_price": max(500, current),
            "pct_remaining":   pct,
        })
    return result

if __name__ == "__main__":
    from database import init_db
    init_db()
    print("\n🚗  Car Price Estimator — http://localhost:5000\n")
    app.run(debug=True, port=5000)