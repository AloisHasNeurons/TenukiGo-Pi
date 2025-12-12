# TenukiGo-Pi (IoT & IaC Migration)

Système d'enregistrement et d'analyse de parties de Go sur Raspberry Pi 3B+, entièrement conteneurisé et piloté par Infrastructure as Code.

![Architecture](https://img.shields.io/badge/Architecture-Ansible%20%2B%20Docker-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%203B%2B-red)

## 🏗 Architecture

Le projet est divisé en deux couches distinctes :

1.  **Infrastructure (Hôte - Ansible)** :
    *   Configuration du système (OS, Optimisations, Docker Engine).
    *   Gestion du Réseau (Bascule WiFi AP / Client).
    *   Gestion matérielle (Daemon python pour les Boutons & LEDs).
    *   Ce daemon pilote le conteneur via `docker exec`.

2.  **Application (Conteneur - Docker)** :
    *   Capture vidéo (`rpicam-vid`).
    *   Logique métier (Découpage, Analyse Sente, IA TFLite).
    *   Encapsulé dans une image `tenukigo-app` basée sur `python:3.10-slim`.

---

## 🛠 Matériel Requis

*   **Raspberry Pi 3B+**.
*   **Raspberry Pi Camera Module 3** (Wide ou Standard).
*   **Interface Physique** :
    *   3 Boutons : **Power**, **Play/Pause**, **Wifi**.
    *   LEDs d'état (RGB / Simples).

---

## 🚀 Installation & Déploiement

### Pré-requis (Sur votre machine de contrôle)
*   Ansible installé.
*   Accès SSH à la Raspberry Pi (IP connue).

### Étape 1 : Configuration Ansible
1.  Editez `ansible/inventory.ini` avec l'IP de votre Pi.
2.  Vérifiez les variables dans `ansible/playbook.yml` (SSID, etc.).

### Étape 2 : Provisioning de l'Infrastructure
Lancez le playbook pour configurer tout le système hôte :
```bash
cd ansible
ansible-playbook -i inventory.ini playbook.yml
```
*Cela installe Docker, configure le WiFi, et lance le service de gestion des boutons.*

### Étape 3 : Déploiement de l'Application (Via GitHub Releases)
L'image Docker est pré-compilée et disponible sur GitHub. Sur la Raspberry Pi :

```bash
# 1. Télécharger l'archive de la dernière release
wget https://github.com/AloisHasNeurons/TenukiGo-Pi/releases/download/v1.0.0-alpha/tenukigo-app.tar

# 2. Charger l'image dans Docker
docker load -i tenukigo-app.tar

# 3. Démarrer le conteneur
docker run -d \
  --name tenukigo-app \
  --privileged \
  --restart unless-stopped \
  -v /home/pi/videos:/app/go_videos \
  -v /home/pi/sgf:/app/output_sgf \
  tenukigo-app:latest
```

*(Note : Si vous souhaitez modifier le code, vous pouvez toujours reconstruire l'image localement via `podman build` ou `docker build`)*

---

## 🎮 Utilisation

### Interface Physique
*   **Bouton Power** : Gestion de l'alimentation.
*   **Bouton Play/Pause** : Lance ou met en pause l'enregistrement/analyse.
*   **Bouton Wifi** : Bascule entre mode **Access Point** (LED Bleue) et **Client** (LED Verte).

### Récupération des Parties
Les fichiers `.sgf` sont générés dans le dossier monté `/app/output_sgf`. Vous pouvez les récupérer via SCP.

---

## 📂 Structure du Projet

```text
.
├── ansible/            # IaC : Configuration du système hôte
│   └── roles/
│       ├── buttons/    # Service systemd pour les boutons
│       └── network/    # Scripts de gestion WiFi
├── app/                # Code source de l'application
│   ├── Dockerfile      # Image technique (Python + RPi Libs)
│   ├── main.py         # Point d'entrée de l'analyse
│   ├── scripts/        # Scripts bash (capture vidéo)
│   └── src/            # Logique Python (IA, SGF)
└── models/             # Modèles TFLite / YOLO
```