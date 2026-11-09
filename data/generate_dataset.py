"""
MECHA Industrial AI Project
============================
Script de génération du jeu de données synthétique
Simule des capteurs IoT sur 5 usines (3 France, 2 Espagne)
Cas d'usage : Maintenance Prédictive

Auteurs : Équipe MSPR TPRE841
Date    : 2025-2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ─── Paramètres de simulation ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

USINES = {
    "USN-FR-01": {"pays": "France",  "ville": "Toulouse",  "nb_machines": 8},
    "USN-FR-02": {"pays": "France",  "ville": "Lyon",      "nb_machines": 6},
    "USN-FR-03": {"pays": "France",  "ville": "Bordeaux",  "nb_machines": 7},
    "USN-ES-01": {"pays": "Espagne", "ville": "Barcelone", "nb_machines": 5},
    "USN-ES-02": {"pays": "Espagne", "ville": "Valence",   "nb_machines": 4},
}

TYPE_MACHINES = ["CNC-Fraisage", "CNC-Tournage", "Découpe-Laser", "Centre-Usinage"]

# Durée de simulation : 18 mois (granularité 1 heure)
DATE_DEBUT = datetime(2024, 1, 1)
DATE_FIN   = datetime(2025, 6, 30)

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def age_machine_aleatoire():
    """Retourne un âge machine en heures (entre 1 000 et 50 000h)."""
    return random.randint(1000, 50000)

def simuler_degradation(heures_restantes_avant_panne, bruit=True):
    """
    Calcule un score de dégradation [0, 1].
    Score proche de 1 → dégradation avancée (panne imminente).
    """
    score = 1 - (heures_restantes_avant_panne / 720)  # horizon 720h (30 jours)
    score = np.clip(score, 0, 1)
    if bruit:
        score += np.random.normal(0, 0.03)
        score = np.clip(score, 0, 1)
    return round(score, 4)

def simuler_capteurs(score_degradation, type_machine, mode_panne=False):
    """
    Génère des valeurs capteurs cohérentes avec le niveau de dégradation.
    Paramètres différents selon le type de machine.
    """
    # Valeurs nominales par type de machine
    config = {
        "CNC-Fraisage":     {"temp_base": 65, "vib_base": 2.1, "courant_base": 18},
        "CNC-Tournage":     {"temp_base": 60, "vib_base": 1.8, "courant_base": 15},
        "Découpe-Laser":    {"temp_base": 45, "vib_base": 0.8, "courant_base": 22},
        "Centre-Usinage":   {"temp_base": 70, "vib_base": 2.5, "courant_base": 25},
    }
    cfg = config[type_machine]

    # Dérive progressive des capteurs avec la dégradation
    facteur = 1 + score_degradation * 0.45
    if mode_panne:
        facteur *= 1.35  # surchauffe / surcharge en approche de panne

    temperature_C   = cfg["temp_base"]   * facteur + np.random.normal(0, 1.5)
    vibration_mm_s  = cfg["vib_base"]    * facteur + np.random.normal(0, 0.15)
    courant_A       = cfg["courant_base"] * facteur + np.random.normal(0, 0.8)
    pression_bar    = round(6.5 - score_degradation * 1.2 + np.random.normal(0, 0.1), 2)
    vitesse_tr_min  = round(max(100, 1500 - score_degradation * 400 + np.random.normal(0, 30)), 0)

    return {
        "temperature_C":   round(temperature_C, 1),
        "vibration_mm_s":  round(vibration_mm_s, 3),
        "courant_A":       round(courant_A, 2),
        "pression_bar":    round(pression_bar, 2),
        "vitesse_tr_min":  int(vitesse_tr_min),
    }

def generer_machine(usine_id, machine_id_local, type_machine, dates):
    """
    Génère la série temporelle complète pour une machine.
    Inclut des cycles de vie avec pannes planifiées / non planifiées.
    """
    records = []
    age_initial = age_machine_aleatoire()

    # Planification des pannes sur la période
    # En moyenne 1 panne tous les 2 mois (≈ 1440h) avec variabilité
    duree_totale_h = len(dates)
    pannes_planifiees = []
    t = random.randint(300, 1200)
    while t < duree_totale_h:
        duree_panne = random.randint(4, 48)   # 4 à 48h d'arrêt
        pannes_planifiees.append((t, duree_panne))
        intervalle_suivant = int(np.random.normal(1440, 300))
        intervalle_suivant = max(500, intervalle_suivant)
        t += intervalle_suivant + duree_panne

    # Dernière maintenance (en heures avant le début)
    h_depuis_maintenance = random.randint(0, 600)

    en_panne = False
    nb_pannes = 0
    nb_maintenances = 0

    for i, date in enumerate(dates):
        age_courant = age_initial + i

        # Vérifier début / fin de panne
        panne_active = any(debut <= i < debut + duree for debut, duree in pannes_planifiees)

        if panne_active and not en_panne:
            en_panne = True
            nb_pannes += 1
            # La panne génère une maintenance corrective
        elif not panne_active and en_panne:
            en_panne = False
            h_depuis_maintenance = 0  # maintenance corrective effectuée
            nb_maintenances += 1

        # Heures restantes avant la prochaine panne
        prochaine_panne = None
        for debut, duree in pannes_planifiees:
            if debut > i:
                prochaine_panne = debut - i
                break
        rul = prochaine_panne if prochaine_panne is not None else 720  # RUL : Remaining Useful Life (h)
        rul = min(rul, 720)  # Plafonné à 30 jours
        if en_panne:
            rul = 0  # Machine déjà en panne : RUL = 0

        score_deg = simuler_degradation(rul)
        capteurs = simuler_capteurs(score_deg, type_machine, mode_panne=(rul < 48))

        # Étiquette de classification
        if en_panne:
            etat = "en_panne"
        elif rul < 48:
            etat = "critique"     # Panne dans moins de 48h
        elif rul < 168:
            etat = "a_risque"     # Panne dans moins d'1 semaine
        else:
            etat = "normal"

        h_depuis_maintenance += 1

        records.append({
            "timestamp":                date.strftime("%Y-%m-%d %H:%M:%S"),
            "usine_id":                 usine_id,
            "machine_id":               f"{usine_id}-M{machine_id_local:02d}",
            "type_machine":             type_machine,
            "age_machine_h":            age_courant,
            "h_depuis_maintenance":     h_depuis_maintenance,
            "temperature_C":            capteurs["temperature_C"],
            "vibration_mm_s":           capteurs["vibration_mm_s"],
            "courant_A":                capteurs["courant_A"],
            "pression_bar":             capteurs["pression_bar"],
            "vitesse_tr_min":           capteurs["vitesse_tr_min"],
            "score_degradation":        score_deg,
            "rul_heures":               rul,       # Remaining Useful Life
            "etat_machine":             etat,
            "en_panne":                 int(en_panne),
            "nb_pannes_total":          nb_pannes,
            "nb_maintenances_total":    nb_maintenances,
        })

    return records

# ─── Génération principale ────────────────────────────────────────────────────

def generer_dataset():
    print("=" * 60)
    print("  MECHA — Génération du jeu de données synthétique")
    print("=" * 60)

    # Génération des timestamps (toutes les heures)
    dates = [DATE_DEBUT + timedelta(hours=h)
             for h in range(int((DATE_FIN - DATE_DEBUT).total_seconds() // 3600))]
    print(f"Période    : {DATE_DEBUT.date()} → {DATE_FIN.date()}")
    print(f"Timestamps : {len(dates):,} points / machine\n")

    all_records = []

    for usine_id, infos in USINES.items():
        print(f"  Usine {usine_id} ({infos['ville']}, {infos['pays']}) — {infos['nb_machines']} machines")
        for m in range(1, infos["nb_machines"] + 1):
            type_m = random.choice(TYPE_MACHINES)
            records = generer_machine(usine_id, m, type_m, dates)
            all_records.extend(records)

    df = pd.DataFrame(all_records)
    print(f"\nDataset total : {len(df):,} lignes × {len(df.columns)} colonnes")

    # Sauvegarde
    os.makedirs(".", exist_ok=True)
    df.to_csv("mecha_dataset_full.csv", index=False)
    print("  ✓ mecha_dataset_full.csv sauvegardé")

    # Échantillon allégé (une mesure toutes les 4h) pour démonstrations
    df_sample = df[df.index % 4 == 0].reset_index(drop=True)
    df_sample.to_csv("mecha_dataset_sample.csv", index=False)
    print(f"  ✓ mecha_dataset_sample.csv sauvegardé ({len(df_sample):,} lignes)")

    # Distribution des états
    print("\nDistribution des états machines :")
    print(df["etat_machine"].value_counts().to_string())
    print(f"\nTaux de panne global : {df['en_panne'].mean()*100:.2f}%")

    return df

if __name__ == "__main__":
    df = generer_dataset()
    print("\n✅ Génération terminée avec succès !")
