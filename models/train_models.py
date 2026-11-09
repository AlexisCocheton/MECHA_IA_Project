"""
MECHA Industrial AI Project
============================
Entraînement des modèles de Machine Learning

Modèle 1 : Classification multi-classes de l'état machine
          (normal / a_risque / critique / en_panne)
          → Random Forest Classifier

Modèle 2 : Régression — Prédiction du RUL (Remaining Useful Life)
          → Random Forest Regressor + comparaison Ridge Regression

Auteurs : Équipe MSPR TPRE841
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Chemins ─────────────────────────────────────────────────────────────────
DATA_PATH   = "../data/mecha_dataset_full.csv"
OUTPUT_DIR  = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Features utilisées ───────────────────────────────────────────────────────
FEATURES = [
    "age_machine_h",
    "h_depuis_maintenance",
    "temperature_C",
    "vibration_mm_s",
    "courant_A",
    "pression_bar",
    "vitesse_tr_min",
]
TARGET_CLF = "etat_machine"
TARGET_RUL = "rul_heures"

print("=" * 60)
print("  MECHA — Entraînement des modèles ML")
print("=" * 60)

# ─── Chargement des données ───────────────────────────────────────────────────
print("\n[1/6] Chargement des données...")
df = pd.read_csv(DATA_PATH)
print(f"  Dataset : {len(df):,} lignes")
print(f"  Features : {FEATURES}")

# Sous-échantillonnage stratifié pour l'entraînement (1 mesure / 4h)
df_train = df[df.index % 4 == 0].reset_index(drop=True)
print(f"  Sous-échantillon : {len(df_train):,} lignes (granularité 4h)")

X = df_train[FEATURES]
y_clf = df_train[TARGET_CLF]
y_rul = df_train[TARGET_RUL]

# ─── Encodage de la cible classification ─────────────────────────────────────
le = LabelEncoder()
y_clf_enc = le.fit_transform(y_clf)
print(f"\n  Classes : {list(le.classes_)}")
print(f"  Distribution :\n{y_clf.value_counts().to_string()}")

# ─── Split train / test (80/20 — respecte la chronologie par usine) ──────────
print("\n[2/6] Split train/test 80/20...")
X_train, X_test, y_clf_train, y_clf_test, y_rul_train, y_rul_test = train_test_split(
    X, y_clf_enc, y_rul, test_size=0.2, random_state=42, stratify=y_clf_enc
)
print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

# ─── MODÈLE 1 : Random Forest Classifier ─────────────────────────────────────
print("\n[3/6] Entraînement — Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_clf_train)
y_clf_pred = clf.predict(X_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print("\n  Rapport de classification :")
print(classification_report(y_clf_test, y_clf_pred, target_names=le.classes_))

# Rapport détaillé (dict) pour métriques JSON
clf_report_dict = classification_report(
    y_clf_test, y_clf_pred, target_names=le.classes_, output_dict=True
)

# Métriques sur l'ensemble d'entraînement (détection de sur-apprentissage)
acc_train = accuracy_score(y_clf_train, clf.predict(X_train))
print(f"  Train Accuracy : {acc_train:.4f} | Test : {acc:.4f} | Gap : {(acc_train - acc)*100:+.2f}pp")

# Validation croisée 5-fold
cv_acc = cross_val_score(clf, X_train, y_clf_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"  CV Accuracy (5-fold) : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# Matrice de confusion
cm = confusion_matrix(y_clf_test, y_clf_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title("Matrice de confusion — Classification état machine\n(Random Forest)", fontsize=12)
ax.set_ylabel("Réel"); ax.set_xlabel("Prédit")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_clf.png"), dpi=150)
plt.close()
print("  ✓ confusion_matrix_clf.png sauvegardée")

# Importance des features — classification
fi_clf = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
fi_clf.plot(kind="barh", color="steelblue", ax=ax)
ax.set_title("Importance des variables — Classifier", fontsize=12)
ax.set_xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_clf.png"), dpi=150)
plt.close()
print("  ✓ feature_importance_clf.png sauvegardée")

# ─── MODÈLE 2 : Random Forest Regressor — RUL ────────────────────────────────
print("\n[4/6] Entraînement — Random Forest Regressor (RUL)...")
reg_rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
reg_rf.fit(X_train, y_rul_train)
y_rul_pred_rf = reg_rf.predict(X_test)
mae_rf  = mean_absolute_error(y_rul_test, y_rul_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_rul_test, y_rul_pred_rf))
r2_rf   = r2_score(y_rul_test, y_rul_pred_rf)
print(f"  RF  — MAE : {mae_rf:.2f}h | RMSE : {rmse_rf:.2f}h | R² : {r2_rf:.4f}")

# Métriques sur l'ensemble d'entraînement
y_rul_train_pred_rf = reg_rf.predict(X_train)
mae_rf_train = mean_absolute_error(y_rul_train, y_rul_train_pred_rf)
r2_rf_train  = r2_score(y_rul_train, y_rul_train_pred_rf)
print(f"  RF  — Train MAE : {mae_rf_train:.2f}h | R² : {r2_rf_train:.4f}")

# Validation croisée 5-fold
cv_mae_rf = -cross_val_score(
    reg_rf, X_train, y_rul_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
)
print(f"  RF  — CV MAE (5-fold) : {cv_mae_rf.mean():.2f}h ± {cv_mae_rf.std():.2f}h")

# Comparaison : Ridge Regression (modèle de référence)
print("\n[5/6] Comparaison — Ridge Regression (baseline)...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
reg_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
reg_ridge.fit(X_train_sc, y_rul_train)
y_rul_pred_ridge = reg_ridge.predict(X_test_sc)
mae_ridge  = mean_absolute_error(y_rul_test, y_rul_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_rul_test, y_rul_pred_ridge))
r2_ridge   = r2_score(y_rul_test, y_rul_pred_ridge)
print(f"  Ridge — MAE : {mae_ridge:.2f}h | RMSE : {rmse_ridge:.2f}h | R² : {r2_ridge:.4f}")

# Graphique comparaison RUL prédit vs réel
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, titre, mae, r2 in zip(
    axes,
    [y_rul_pred_rf, y_rul_pred_ridge],
    ["Random Forest", "Ridge (baseline)"],
    [mae_rf, mae_ridge],
    [r2_rf, r2_ridge],
):
    n_sample = min(1000, len(y_rul_test))
    sample = np.random.choice(len(y_rul_test), n_sample, replace=False)
    ax.scatter(y_rul_test.iloc[sample], y_pred[sample], alpha=0.3, s=10, color="steelblue")
    lim = [0, 730]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Prédiction parfaite")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RUL réel (h)"); ax.set_ylabel("RUL prédit (h)")
    ax.set_title(f"{titre}\nMAE={mae:.1f}h | R²={r2:.3f}")
    ax.legend()
plt.suptitle("Prédiction du RUL (Remaining Useful Life) — Comparaison modèles", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rul_prediction_comparison.png"), dpi=150)
plt.close()
print("  ✓ rul_prediction_comparison.png sauvegardée")

# Importance features — régression
fi_reg = pd.Series(reg_rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
fi_reg.plot(kind="barh", color="darkorange", ax=ax)
ax.set_title("Importance des variables — Régression RUL", fontsize=12)
ax.set_xlabel("Importance (MSE)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_reg.png"), dpi=150)
plt.close()
print("  ✓ feature_importance_reg.png sauvegardée")

# ─── Sauvegarde des modèles ───────────────────────────────────────────────────
print("\n[6/6] Sauvegarde des modèles...")
joblib.dump(clf,       os.path.join(OUTPUT_DIR, "model_classifier.pkl"))
joblib.dump(reg_rf,    os.path.join(OUTPUT_DIR, "model_rul_regressor.pkl"))
joblib.dump(le,        os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler,    os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(reg_ridge, os.path.join(OUTPUT_DIR, "model_rul_ridge_baseline.pkl"))
print("  ✓ model_classifier.pkl")
print("  ✓ model_rul_regressor.pkl")
print("  ✓ label_encoder.pkl")
print("  ✓ scaler.pkl")
print("  ✓ model_rul_ridge_baseline.pkl")

# Métriques JSON
metrics = {
    "classifier": {
        "modele": "RandomForestClassifier",
        "hyperparametres": {
            "n_estimators": 150, "max_depth": 12,
            "min_samples_leaf": 5, "class_weight": "balanced"
        },
        "accuracy":          round(acc, 4),
        "accuracy_train":    round(acc_train, 4),
        "cv_accuracy_mean":  round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":   round(float(cv_acc.std()), 4),
        "per_class_metrics": {
            cls: {
                "precision": round(clf_report_dict[cls]["precision"], 4),
                "recall":    round(clf_report_dict[cls]["recall"], 4),
                "f1-score":  round(clf_report_dict[cls]["f1-score"], 4),
            }
            for cls in le.classes_
        },
        "classes":  list(le.classes_),
        "features": FEATURES,
    },
    "rul_regressor": {
        "modele": "RandomForestRegressor",
        "hyperparametres": {
            "n_estimators": 150, "max_depth": 12,
            "min_samples_leaf": 5
        },
        "MAE_heures":       round(mae_rf, 2),
        "RMSE_heures":      round(rmse_rf, 2),
        "R2":               round(r2_rf, 4),
        "MAE_heures_train": round(mae_rf_train, 2),
        "R2_train":         round(r2_rf_train, 4),
        "cv_mae_mean":      round(float(cv_mae_rf.mean()), 2),
        "cv_mae_std":       round(float(cv_mae_rf.std()), 2),
        "features": FEATURES,
    },
    "baseline_ridge": {
        "modele": "RidgeCV",
        "MAE_heures":  round(mae_ridge, 2),
        "RMSE_heures": round(rmse_ridge, 2),
        "R2":          round(r2_ridge, 4),
    },
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("  ✓ metrics.json")

print("\n" + "=" * 60)
print("  RÉSUMÉ DES PERFORMANCES")
print("=" * 60)
print(f"  Classifier (état machine) — Accuracy : {acc*100:.2f}%")
print(f"  Régresseur RUL (RF)       — MAE : {mae_rf:.1f}h | R² : {r2_rf:.3f}")
print(f"  Baseline Ridge            — MAE : {mae_ridge:.1f}h | R² : {r2_ridge:.3f}")
if mae_ridge > mae_rf:
    print(f"\n  → RF surpasse Ridge de {(mae_ridge - mae_rf) / mae_ridge * 100:.1f}% sur le MAE")
elif mae_rf > mae_ridge:
    print(f"\n  → Ridge surpasse RF de {(mae_rf - mae_ridge) / mae_rf * 100:.1f}% sur le MAE (RF sous-optimal)")
else:
    print("\n  → RF et Ridge ont des MAE identiques")
print("\n✅ Entraînement terminé !")
