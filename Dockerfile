# Étape 1: Image de base
FROM python:3.10-slim-bullseye

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 2: Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    libopenjp2-7 \
    libtiff5 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Étape 3: Installer les dépendances Python (sauf 'sente')
COPY requirements.docker.txt .

# On crée un fichier de requirements filtré SANS 'sente'
RUN grep -v '^sente' requirements.docker.txt > requirements.filtered.txt

# On installe pybind11 (dépendance de build/runtime requise par 'sente' mais absente de votre liste)
# Puis on installe toutes les autres dépendances du fichier filtré
RUN pip install --no-cache-dir pybind11 meson ninja && \
    pip install --no-cache-dir -r requirements.filtered.txt

# ÉTAPE 3.5: Patcher et installer 'sente' manuellement
RUN SENTE_SRC_DIR="/app/sente-src" && \
    echo "--- Cloning and patching 'sente' source... ---" && \
    git clone https://github.com/atw1020/sente.git "$SENTE_SRC_DIR" && \
    \
    echo "--- Patching files... ---" && \
    sed -i 's/">=3.8.*"/">=3.8"/' "$SENTE_SRC_DIR/setup.py" && \
    sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/Tree.h" && \
    sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/SGF/SGFNode.cpp" && \
    \
    echo "--- Installing 'sente' from patched source... ---" && \
    # On utilise --no-deps car on a déjà installé les dépendances (numpy, sgf) à l'étape 3
    pip install --no-cache-dir --no-deps "$SENTE_SRC_DIR" && \
    \
    echo "--- Cleaning up 'sente' source... ---" && \
    # On supprime le code source pour garder l'image légère
    rm -rf "$SENTE_SRC_DIR"

# Étape 4: Copier le code de l'application
COPY . .

# Étape 5: Configurer le conteneur
EXPOSE 5000

# Commande pour lancer l'application au démarrage du conteneur
CMD ["python", "main.py"]