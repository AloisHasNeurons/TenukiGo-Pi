# Étape 1: Image de base
# On utilise une image Python 3.10 slim pour 'Bullseye' (Debian 11)
# C'est une image multi-architecture, Docker prendra la version arm/v7 sur le Pi.
FROM python:3.10-slim-bullseye

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 2: Installer les dépendances système minimes
# Nécessaire pour que OpenCV et Torch fonctionnent correctement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    libopenjp2-7 \
    libtiff5 \
    && rm -rf /var/lib/apt/lists/*

# Étape 3: Installer les dépendances Python
# On copie notre nouveau fichier de requirements
COPY requirements.docker.txt .

# On installe les paquets. Cela prendra du temps la première fois.
RUN pip install --no-cache-dir -r requirements.docker.txt

# Étape 4: Copier le code de l'application
# On copie tout le reste du projet dans l'image
COPY . .

# Étape 5: Configurer le conteneur
# Exposer le port 5000 (utilisé par Flask)
EXPOSE 5000

# Commande pour lancer l'application au démarrage du conteneur
CMD ["python", "main.py"]