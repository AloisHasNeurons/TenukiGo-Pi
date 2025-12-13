#!/bin/bash

# --- CONFIGURATION HÔTE (PI) ---
HOST_VIDEO_DIR="/home/mao/go_videos"
HOST_SGF_DIR="/home/mao/output_sgf"

# --- CONFIGURATION DOCKER (CONTENEUR) ---
# Ce sont les chemins VUS par le Docker (via le volume)
DOCKER_CONTAINER_NAME="tenukigo-app"
DOCKER_VIDEO_DIR="/app/go_videos"
DOCKER_SGF_DIR="/app/output_sgf"
DOCKER_MODELS_DIR="/app/models"

mkdir -p "$HOST_VIDEO_DIR"
mkdir -p "$HOST_SGF_DIR"

# Nommage des fichiers
START_TS=$(date +"%Y-%m-%d_%H-%M-%S")
SESSION_PREFIX="tmp_${START_TS}_part_"
FINAL_NAME="go_${START_TS}.h264"
FINAL_SGF="go_${START_TS}.sgf"

echo "=== Début Enregistrement (Hôte) ==="
echo "Fichier : $HOST_VIDEO_DIR/$FINAL_NAME"

finalize() {
  echo ""
  echo "=== Arrêt détecté. Finalisation... ==="
  cd "$HOST_VIDEO_DIR" || exit 1

  # 1. Concaténation (Faite par l'hôte car il a ffmpeg)
  if ls ${SESSION_PREFIX}*.h264 1>/dev/null 2>&1; then
    echo "Concaténation des segments..."
    cat ${SESSION_PREFIX}*.h264 > "$FINAL_NAME"
    rm ${SESSION_PREFIX}*.h264
    echo "Vidéo assemblée : $HOST_VIDEO_DIR/$FINAL_NAME"
  else
    echo "Aucun segment trouvé."
    exit 1
  fi

  # 2. LE LIEN MAGIQUE : On demande à Docker de travailler
  echo "🚀 Envoi de la commande d'analyse au Docker..."
  
  # On utilise 'docker exec' pour lancer la commande DANS le conteneur existant
  # Notez bien l'utilisation des chemins DOCKER_... pour les arguments
  sudo docker exec "$DOCKER_CONTAINER_NAME" python3 /app/main.py \
    --video "$DOCKER_VIDEO_DIR/$FINAL_NAME" \
    --output "$DOCKER_SGF_DIR/$FINAL_SGF" \
    --yolo-model "$DOCKER_MODELS_DIR/model.pt" \
    --keras-model "$DOCKER_MODELS_DIR/modelCNN.tflite" \
    --transparent

  # Vérification du résultat (sur l'hôte, car le volume est partagé)
  if [ -f "$HOST_SGF_DIR/$FINAL_SGF" ]; then
      echo "✅ Succès ! SGF généré : $HOST_SGF_DIR/$FINAL_SGF"
  else
      echo "❌ Erreur : Le SGF n'a pas été créé."
  fi
  
  exit 0
}

trap finalize INT TERM

# Enregistrement (Exécuté par l'hôte)
rpicam-vid --width 1280 --height 720 --framerate 25 --inline -t 0 -o - | \
ffmpeg -re -f h264 -i - -c:v copy -f segment -segment_time 10 \
       -reset_timestamps 1 -segment_format h264 \
       "${HOST_VIDEO_DIR}/${SESSION_PREFIX}%03d.h264"

finalize