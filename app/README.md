# TenukiGo Application

Ceci est le code source de l'application TenukiGo. Il est conçu pour tourner :
1.  **Sur Raspberry Pi** (via Docker + TFLite Runtime).
2.  **Sur PC/Dev** (via Python + TensorFlow/Keras).

## 📂 Structure

*   `Dockerfile` : Définition de l'image de production (Raspberry Pi 3B+ / Arm64).
*   `main.py` : Point d'entrée principal. Analyse une vidéo et génère un SGF.
*   `src/` : Code source du package `tenukigo_pi`.
*   `models/` : Modèles IA (YOLOv8 `.pt` et CNN `.tflite` / `.keras`).
*   `scripts/` : Scripts Bash utilisés par l'image Docker (capture `rpicam-vid`).
*   `lib/` : Roue `sente` compilée (dépendance C++ critique).

---

## 💻 Développement Local (PC/Mac)

Vous pouvez lancer l'analyse sur votre machine sans Docker pour débugger la logique de jeu ou de vision.

### 1. Installation de l'environnement
Le projet utilise **Micromamba** (ou Conda) pour gérer les dépendances complexes (TensorFlow, PyTorch, OpenCV).

```bash
# 1. Aller dans le dossier de l'application
cd app

# 2. Créer l'environnement à partir du fichier environment.yml
micromamba env create -f environment.yml

# 3. Activer l'environnement
micromamba activate tenukigo_pi

# 4. Installer le package local en mode éditable
# Cela permet de modifier le code dans src/ sans réinstaller
pip install -e .
```
> **Note sur sente** : Si l'installation automatique de la librairie sente échoue (problème de compilation C++ fréquent), vous devrez peut-être installer le wheel pré-compilé manuellement ou suivre les instructions de compilation dans Dockerfile.build_sente.


### 2. Lancer l'analyse
```bash
# Analyse d'une vidéo MP4
python main.py \
  --video data/ma_partie.mp4 \
  --output result.sgf \
  --yolo-model models/model.pt \
  --keras-model models/modelCNN.keras
```

> **Note** : Sur PC, le script utilisera automatiquement TensorFlow/Keras (défini dans environment.yml). Sur Raspberry Pi, il basculera sur TFLite Runtime.

---

## 🐳 Docker (Production RPi)

L'image Docker est optimisée pour la Raspberry Pi (Arm64).

### Build Manuel
```bash
# Depuis le dossier app/
podman build --platform linux/arm64 -t tenukigo-app:latest .
```

### Test Manuel (Sur RPi)
```bash
docker run -it --rm \
  --device /dev/video0 \
  -v $(pwd)/data:/app/go_videos \
  tenukigo-app:latest \
  /bin/bash
```

---

## 📜 Scripts

*   `scripts/cam_go_script.sh` : Script "Chef d'orchestre" lancé par les boutons physiques.
    1.  Lance `rpicam-vid` (enregistrement).
    2.  Attend le signal `SIGINT` (Bouton Stop).
    3.  Lance `main.py` pour analyser la vidéo capturée.