from __future__ import annotations
"""
ml_pipeline.py — Pipeline ML pour AutoEstim
Modèle 1 : Prédiction du prix actuel (XGBoost + GridSearch)
Modèle 2 : Courbe de dépréciation (prix en fonction de l'âge/km)
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Chemins des fichiers
# ─────────────────────────────────────────────

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(_BASE_DIR, "models")
MODEL_PRICE = os.path.join(MODEL_DIR, "price_predictor.pkl")
MODEL_DEPRE = os.path.join(MODEL_DIR, "depreciation.pkl")
META_PATH   = os.path.join(MODEL_DIR, "meta.json")

os.makedirs(MODEL_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year

# ─────────────────────────────────────────────
#  Chargement des données depuis SQLite
# ─────────────────────────────────────────────

def load_data(brand: Optional[str] = None, model: Optional[str] = None) -> pd.DataFrame:
    """Charge les annonces depuis la base SQLite dans un DataFrame."""
    from database import get_listings_for_ml
    rows = get_listings_for_ml(brand=brand, model=model)
    df = pd.DataFrame(rows)
    print(f"  📦 {len(df)} annonces chargées depuis la base.")
    return df

# ─────────────────────────────────────────────
#  Préparation des features
# ─────────────────────────────────────────────

# Mapping carburant → catégorie normalisée
FUEL_MAP = {
    "gasoline": "essence", "petrol": "essence", "essence": "essence", "sp95": "essence",
    "diesel": "diesel", "gazole": "diesel",
    "hybrid": "hybride", "hybride": "hybride", "hybride rechargeable": "hybride",
    "plug-in hybrid": "hybride", "mild hybrid": "hybride",
    "electric": "electrique", "electrique": "electrique", "électrique": "electrique",
    "lpg": "gpl", "gpl": "gpl", "cng": "gpl",
}

TRANS_MAP = {
    "manual": "manuelle", "manuelle": "manuelle", "mécanique": "manuelle",
    "automatic": "automatique", "automatique": "automatique", "auto": "automatique",
    "semi-automatic": "automatique", "dsg": "automatique",
}

def normalize_fuel(s: str) -> str:
    if not s:
        return "autre"
    return FUEL_MAP.get(str(s).lower().strip(), "autre")

def normalize_transmission(s: str) -> str:
    if not s:
        return "inconnue"
    return TRANS_MAP.get(str(s).lower().strip(), "inconnue")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame de features à partir des données brutes.
    Features numériques : age, mileage, power_hp
    Features catégorielles : brand, model, fuel, transmission
    """
    df = df.copy()

    # ── Nettoyage de base ──
    df["price"]   = pd.to_numeric(df["price"],   errors="coerce")
    df["year"]    = pd.to_numeric(df["year"],     errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"],  errors="coerce")
    df["power_hp"]= pd.to_numeric(df.get("power_hp", pd.Series(dtype=float)), errors="coerce")

    # Supprimer les lignes sans prix ou sans année
    df = df.dropna(subset=["price", "year"])
    df = df[df["price"].between(500, 300_000)]
    df = df[df["year"].between(1990, CURRENT_YEAR)]

    # ── Features temporelles ──
    df["age"] = CURRENT_YEAR - df["year"].astype(int)

    # ── Kilométrage : imputation par médiane age×marque ──
    df["mileage"] = df["mileage"].clip(upper=600_000)
    median_km_by_age = df.groupby("age")["mileage"].transform("median")
    df["mileage"] = df["mileage"].fillna(median_km_by_age).fillna(15_000 * df["age"])

    # ── Puissance ──
    df["power_hp"] = df["power_hp"].fillna(df["power_hp"].median())

    # ── Normalisation des catégories ──
    df["fuel"]         = df["fuel"].fillna("").apply(normalize_fuel)
    df["transmission"] = df["transmission"].fillna("").apply(normalize_transmission)
    df["brand"]        = df["brand"].fillna("Autre").str.title()
    df["model"]        = df["model"].fillna("Inconnu")

    return df

def build_X_y(df: pd.DataFrame):
    """Construit les matrices X (features) et y (cible = prix)."""
    from sklearn.preprocessing import OrdinalEncoder

    features_cat = ["brand", "model", "fuel", "transmission"]
    features_num = ["age", "mileage", "power_hp"]

    # Encoder les catégories
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(df[features_cat].astype(str))
    X_num = df[features_num].values
    X = np.hstack([X_cat, X_num])

    y = np.log1p(df["price"].values)  # log-transform pour stabiliser la variance

    return X, y, enc, features_cat + features_num

# ─────────────────────────────────────────────
#  Modèle 1 — Prédiction du prix
# ─────────────────────────────────────────────

def train_price_model(df: pd.DataFrame) -> dict:
    """
    Entraîne un XGBoost avec GridSearchCV pour prédire le prix.
    Retourne les métriques de performance.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("\n🤖 Entraînement du modèle de prix (XGBoost)...")

    df_clean = prepare_features(df)
    print(f"  📊 {len(df_clean)} annonces après nettoyage.")

    if len(df_clean) < 100:
        raise ValueError("Pas assez de données pour entraîner le modèle (minimum 100).")

    X, y, encoder, feature_names = build_X_y(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Grid Search ──
    param_grid = {
        "n_estimators":     [200, 400],
        "max_depth":        [5, 7],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method="hist",   # rapide sur CPU
    )

    print("  🔍 GridSearchCV en cours (patience)...")
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"  ✅ Meilleurs paramètres : {grid.best_params_}")

    # ── Évaluation ──
    y_pred_log = best_model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    mae  = int(mean_absolute_error(y_true, y_pred))
    rmse = int(mean_squared_error(y_true, y_pred, squared=False))
    r2   = round(r2_score(y_true, y_pred), 4)

    print(f"\n  📈 Performances sur le jeu de test :")
    print(f"     MAE  (erreur moyenne)   : {mae:,} €")
    print(f"     RMSE (erreur quadratic) : {rmse:,} €")
    print(f"     R²   (précision)        : {r2} / 1.0")

    # ── Sauvegarde ──
    payload = {
        "model":         best_model,
        "encoder":       encoder,
        "feature_names": feature_names,
    }
    with open(MODEL_PRICE, "wb") as f:
        pickle.dump(payload, f)
    print(f"  💾 Modèle sauvegardé : {MODEL_PRICE}")

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples":  len(df_clean),
        "mae":        mae,
        "rmse":       rmse,
        "r2":         r2,
        "best_params": grid.best_params_,
    }

    # Log dans la base
    _log_ml_run("price_predictor", len(df_clean), mae, rmse, r2)

    return metrics

# ─────────────────────────────────────────────
#  Modèle 2 — Courbe de dépréciation
# ─────────────────────────────────────────────

def train_depreciation_model(df: pd.DataFrame) -> dict:
    """
    Modélise la dépréciation en calculant le prix médian
    par (age, tranche_km) pour chaque marque/modèle.
    Complété par une régression polynomiale globale.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    print("\n📉 Entraînement du modèle de dépréciation...")

    df_clean = prepare_features(df)

    # ── Courbe globale : prix en fonction de l'âge ──
    # On normalise par le prix médian de l'année 0 pour avoir un % de valeur résiduelle
    df_ref = df_clean.copy()
    df_ref["age_km"] = df_ref["age"] + df_ref["mileage"] / 150_000  # facteur combiné

    X_dep = df_ref[["age", "mileage"]].values
    y_dep = df_ref["price"].values / df_ref.groupby("brand")["price"].transform("max").clip(lower=1)
    y_dep = np.clip(y_dep, 0.01, 1.5)

    model_dep = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=1.0),
    )
    model_dep.fit(X_dep, y_dep)

    # ── Table de dépréciation par marque/modèle ──
    depre_table = (
        df_clean.groupby(["brand", "model", "age"])["price"]
        .agg(["median", "count"])
        .reset_index()
    )
    depre_table = depre_table[depre_table["count"] >= 3]
    depre_table.columns = ["brand", "model", "age", "median_price", "count"]

    payload = {
        "global_model":  model_dep,
        "depre_table":   depre_table.to_dict("records"),
    }
    with open(MODEL_DEPRE, "wb") as f:
        pickle.dump(payload, f)
    print(f"  💾 Modèle de dépréciation sauvegardé : {MODEL_DEPRE}")
    print(f"  📊 {len(depre_table)} points de dépréciation calculés.")

    _log_ml_run("depreciation", len(df_clean), None, None, None)
    return {"n_points": len(depre_table)}

# ─────────────────────────────────────────────
#  Inférence — Prédiction d'un prix
# ─────────────────────────────────────────────

def predict_price(
    brand: str,
    model: str,
    year: int,
    mileage: int,
    fuel: str = "",
    transmission: str = "",
    power_hp: Optional[float] = None,
) -> Optional[dict]:
    """
    Prédit le prix d'une voiture avec le modèle entraîné.
    Retourne un dict avec la prédiction et un intervalle de confiance.
    """
    if not os.path.exists(MODEL_PRICE):
        return None

    with open(MODEL_PRICE, "rb") as f:
        payload = pickle.load(f)

    xgb_model = payload["model"]
    encoder   = payload["encoder"]

    age = CURRENT_YEAR - int(year)
    if power_hp is None:
        power_hp = 110.0  # valeur par défaut raisonnable

    fuel_norm  = normalize_fuel(fuel)
    trans_norm = normalize_transmission(transmission)

    X_cat = encoder.transform([[brand, model, fuel_norm, trans_norm]])
    X_num = np.array([[age, mileage, power_hp]])
    X = np.hstack([X_cat, X_num])

    log_pred = xgb_model.predict(X)[0]
    price    = int(np.expm1(log_pred))

    # Intervalle de confiance : ±15% du prix prédit
    margin = int(price * 0.15)
    low    = max(0, price - margin)
    high   = price + margin

    return {
        "predicted_price": price,
        "confidence_low":  low,
        "confidence_high": high,
        "age":             age,
    }

# ─────────────────────────────────────────────
#  Inférence — Courbe de dépréciation
# ─────────────────────────────────────────────

def predict_depreciation(
    brand: str,
    model: str,
    current_year: int,
    current_mileage: int,
    current_price: int,
    years_ahead: int = 10,
) -> list[dict]:
    """
    Prédit la valeur de la voiture pour les N prochaines années.
    Retourne une liste de {year, age, predicted_price, pct_remaining}.
    """
    if not os.path.exists(MODEL_DEPRE):
        return _fallback_depreciation(current_price, current_year, years_ahead)

    with open(MODEL_DEPRE, "rb") as f:
        payload = pickle.load(f)

    global_model = payload["global_model"]
    depre_table  = pd.DataFrame(payload["depre_table"])

    current_age = CURRENT_YEAR - int(current_year)
    km_per_year = 15_000   # km moyen par an

    # Cherche la table de dépré spécifique à la marque/modèle
    mask = (
        (depre_table["brand"].str.lower() == brand.lower()) &
        (depre_table["model"].str.lower() == model.lower())
    )
    specific = depre_table[mask].sort_values("age")

    curve = []
    for delta in range(0, years_ahead + 1):
        future_age = current_age + delta
        future_km  = current_mileage + delta * km_per_year
        future_year_cal = CURRENT_YEAR + delta

        # Priorité : table spécifique marque/modèle
        specific_match = specific[specific["age"] == future_age]
        if not specific_match.empty:
            pred_price = int(specific_match.iloc[0]["median_price"])
        else:
            # Fallback : modèle global (régression polynomiale)
            X_dep = np.array([[future_age, future_km]])
            pct   = float(global_model.predict(X_dep)[0])
            pct   = np.clip(pct, 0.01, 1.5)
            pred_price = int(current_price * pct)

        pct_remaining = round(pred_price / current_price * 100, 1) if current_price > 0 else 0

        curve.append({
            "calendar_year":  future_year_cal,
            "age":            future_age,
            "mileage_est":    future_km,
            "predicted_price": pred_price,
            "pct_remaining":  pct_remaining,
        })

    return curve

def _fallback_depreciation(price: int, year: int, years_ahead: int) -> list[dict]:
    """Courbe de dépréciation empirique si le modèle n'est pas encore entraîné."""
    # Taux de dépréciation annuel typique : ~15% les 3 premières années, ~8% ensuite
    current_age = CURRENT_YEAR - int(year)
    curve = []
    current_val = price
    for delta in range(0, years_ahead + 1):
        age = current_age + delta
        rate = 0.15 if age <= 3 else (0.10 if age <= 6 else 0.07)
        if delta > 0:
            current_val = int(current_val * (1 - rate))
        curve.append({
            "calendar_year":   CURRENT_YEAR + delta,
            "age":             age,
            "mileage_est":     15_000 * (current_age + delta),
            "predicted_price": max(500, current_val),
            "pct_remaining":   round(max(500, current_val) / price * 100, 1),
        })
    return curve

# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────

def _log_ml_run(model_type: str, n_samples: int, mae, rmse, r2):
    """Enregistre un run ML dans la base."""
    try:
        from database import get_connection
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO ml_runs (run_at, model_type, n_samples, mae, rmse, r2)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), model_type, n_samples, mae, rmse, r2),
            )
    except Exception:
        pass

def is_model_trained() -> bool:
    """Vérifie si les modèles sont disponibles."""
    return os.path.exists(MODEL_PRICE) and os.path.exists(MODEL_DEPRE)

def get_model_meta() -> dict:
    """Retourne les métadonnées du dernier entraînement."""
    try:
        from database import get_connection
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ml_runs ORDER BY run_at DESC LIMIT 2"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []

# ─────────────────────────────────────────────
#  Point d'entrée — Entraînement complet
# ─────────────────────────────────────────────

def train_all():
    """Lance l'entraînement complet des deux modèles."""
    print("\n" + "═" * 50)
    print("  🚀 AutoEstim — Pipeline ML")
    print("═" * 50)

    df = load_data()

    if df.empty:
        print("❌ Aucune donnée en base. Lance d'abord un import CSV.")
        return

    # Modèle 1 : prix
    metrics_price = train_price_model(df)

    # Modèle 2 : dépréciation
    metrics_depre = train_depreciation_model(df)

    print("\n" + "═" * 50)
    print("  ✅ Entraînement terminé !")
    print(f"     MAE  : {metrics_price['mae']:,} € (erreur moyenne)")
    print(f"     R²   : {metrics_price['r2']} / 1.0")
    print("═" * 50)

if __name__ == "__main__":
    train_all()