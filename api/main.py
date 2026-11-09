"""
MECHA Industrial AI — API REST (FastAPI)
=========================================
Expose les modèles ML de maintenance prédictive via une API REST.

Endpoints :
  GET  /              → accueil
  GET  /health        → statut de l'API
  POST /predict       → prédiction état machine + RUL
  POST /predict/batch → prédiction sur un lot de machines
  GET  /metrics       → métriques des modèles

Lancement :
  uvicorn main:app --host 0.0.0.0 --port 8000

Auteurs : Équipe MSPR TPRE841
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import json
import os

# ─── Chemins des modèles ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")

# ─── Chargement des modèles ───────────────────────────────────────────────────
try:
    clf    = joblib.load(os.path.join(MODELS_DIR, "model_classifier.pkl"))
    reg    = joblib.load(os.path.join(MODELS_DIR, "model_rul_regressor.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
        METRICS = json.load(f)
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)

# ─── Application FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="MECHA Predictive Maintenance API",
    description=(
        "API de maintenance prédictive pour le système industriel MECHA. "
        "Prédit l'état des machines et le temps restant avant panne (RUL)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class MachineSensorData(BaseModel):
    machine_id:           str   = Field(...,  example="USN-FR-01-M01")
    age_machine_h:        float = Field(...,  ge=0,    example=15000)
    h_depuis_maintenance: float = Field(...,  ge=0,    example=320)
    temperature_C:        float = Field(...,  example=72.4)
    vibration_mm_s:       float = Field(...,  ge=0,    example=2.3)
    courant_A:            float = Field(...,  example=18.5)
    pression_bar:         float = Field(...,  example=6.1)
    vitesse_tr_min:       float = Field(...,  ge=0,    example=1420)

class PredictionResult(BaseModel):
    machine_id:        str
    etat_predit:       str
    probabilites:      dict
    rul_predit_heures: float
    score_risque:      float
    recommandation:    str
    alerte:            bool

class BatchRequest(BaseModel):
    machines: List[MachineSensorData] = Field(..., max_length=100)

# ─── Helpers ──────────────────────────────────────────────────────────────────

FEATURES = [
    "age_machine_h", "h_depuis_maintenance",
    "temperature_C", "vibration_mm_s",
    "courant_A", "pression_bar", "vitesse_tr_min",
]

RECOMMANDATIONS = {
    "normal":   "Aucune action requise. Surveillance standard.",
    "a_risque": "Planifier une maintenance dans les 7 jours.",
    "critique": "Maintenance urgente sous 48h. Alerter le responsable.",
    "en_panne": "Machine arrêtée. Intervention immédiate nécessaire.",
}

def make_prediction(data: MachineSensorData) -> PredictionResult:
    """Applique les deux modèles et construit la réponse."""
    X = np.array([[
        data.age_machine_h,
        data.h_depuis_maintenance,
        data.temperature_C,
        data.vibration_mm_s,
        data.courant_A,
        data.pression_bar,
        data.vitesse_tr_min,
    ]])

    # Classification
    etat_enc = clf.predict(X)[0]
    etat = le.inverse_transform([etat_enc])[0]
    proba_raw = clf.predict_proba(X)[0]
    probabilites = {cls: round(float(p), 4)
                    for cls, p in zip(le.classes_, proba_raw)}

    # RUL
    rul = float(reg.predict(X)[0])
    rul = max(0.0, min(720.0, rul))

    # Score de risque [0–1]
    score = probabilites.get("critique", 0) + probabilites.get("en_panne", 0)
    score = round(min(1.0, score + probabilites.get("a_risque", 0) * 0.4), 4)

    alerte = etat in ("critique", "en_panne") or rul < 48

    return PredictionResult(
        machine_id        = data.machine_id,
        etat_predit       = etat,
        probabilites      = probabilites,
        rul_predit_heures = round(rul, 1),
        score_risque      = score,
        recommandation    = RECOMMANDATIONS[etat],
        alerte            = alerte,
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def accueil():
    return {
        "service": "MECHA Predictive Maintenance API",
        "version": "1.0.0",
        "statut":  "opérationnel" if MODELS_LOADED else "erreur chargement modèles",
        "docs":    "/docs",
    }

@app.get("/health", tags=["Info"])
def health():
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail=f"Modèles non chargés : {LOAD_ERROR}")
    return {"statut": "ok", "modeles_charges": True}

@app.post("/predict", response_model=PredictionResult, tags=["Prédiction"])
def predict(data: MachineSensorData):
    """
    Prédit l'état d'une machine et son RUL à partir des mesures capteurs.
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Modèles non disponibles")
    return make_prediction(data)

@app.post("/predict/batch", response_model=List[PredictionResult], tags=["Prédiction"])
def predict_batch(request: BatchRequest):
    """
    Prédit l'état et le RUL pour un lot de machines en une seule requête.
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Modèles non disponibles")
    return [make_prediction(m) for m in request.machines]

@app.get("/metrics", tags=["Info"])
def get_metrics():
    """
    Retourne les métriques de performance des modèles entraînés.
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Modèles non disponibles")
    return METRICS
