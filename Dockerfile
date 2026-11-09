# ─── MECHA Predictive Maintenance — Image Docker ─────────────────────────────
# Image de base légère Python 3.11
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Equipe MSPR TPRE841"
LABEL description="MECHA — API de maintenance prédictive IA"
LABEL version="1.0.0"

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY api/       ./api/
COPY models/    ./models/

# Exposition du port API
EXPOSE 8000

# Variable d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
