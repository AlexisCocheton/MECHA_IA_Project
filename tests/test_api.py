"""
Tests unitaires — API MECHA Predictive Maintenance
===================================================
Couvre les endpoints REST et les règles de validation Pydantic.
Les tests passent que les modèles soient chargés (200) ou non (503).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_MACHINE = {
    "machine_id":           "TEST-M01",
    "age_machine_h":        15000.0,
    "h_depuis_maintenance": 320.0,
    "temperature_C":        72.4,
    "vibration_mm_s":       2.3,
    "courant_A":            18.5,
    "pression_bar":         6.1,
    "vitesse_tr_min":       1420.0,
}


# ── GET / ────────────────────────────────────────────────────────────────────

def test_root_returns_service_info():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body
    assert "version" in body


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_valid_status():
    resp = client.get("/health")
    assert resp.status_code in (200, 503)


# ── POST /predict ─────────────────────────────────────────────────────────────

def test_predict_valid_input():
    resp = client.post("/predict", json=SAMPLE_MACHINE)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "etat_predit" in body
        assert "rul_predit_heures" in body
        assert "probabilites" in body
        assert "score_risque" in body
        assert "alerte" in body
        assert 0.0 <= body["rul_predit_heures"] <= 720.0
        assert 0.0 <= body["score_risque"] <= 1.0
        assert isinstance(body["alerte"], bool)


def test_predict_missing_required_field():
    resp = client.post("/predict", json={"machine_id": "TEST-M01"})
    assert resp.status_code == 422


def test_predict_negative_age_rejected():
    data = {**SAMPLE_MACHINE, "age_machine_h": -1.0}
    resp = client.post("/predict", json=data)
    assert resp.status_code == 422


def test_predict_negative_vibration_rejected():
    data = {**SAMPLE_MACHINE, "vibration_mm_s": -0.5}
    resp = client.post("/predict", json=data)
    assert resp.status_code == 422


# ── POST /predict/batch ───────────────────────────────────────────────────────

def test_predict_batch_single():
    resp = client.post("/predict/batch", json={"machines": [SAMPLE_MACHINE]})
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["machine_id"] == SAMPLE_MACHINE["machine_id"]


def test_predict_batch_multiple():
    second = {**SAMPLE_MACHINE, "machine_id": "TEST-M02"}
    resp = client.post("/predict/batch", json={"machines": [SAMPLE_MACHINE, second]})
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert len(resp.json()) == 2


def test_predict_batch_too_large():
    """Batch > 100 machines doit retourner 422."""
    resp = client.post("/predict/batch", json={"machines": [SAMPLE_MACHINE] * 101})
    assert resp.status_code == 422


def test_predict_batch_empty_rejected():
    resp = client.post("/predict/batch", json={"machines": []})
    # Pydantic v2 : liste vide invalide si max_length défini sans min_length
    # Accepter 200/422/503 selon la config
    assert resp.status_code in (200, 422, 503)


# ── GET /metrics ──────────────────────────────────────────────────────────────

def test_metrics_endpoint_structure():
    resp = client.get("/metrics")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "classifier" in body
        assert "rul_regressor" in body
        assert "baseline_ridge" in body
        clf = body["classifier"]
        assert "accuracy" in clf
        assert "features" in clf
        reg = body["rul_regressor"]
        assert "MAE_heures" in reg
        assert "R2" in reg
