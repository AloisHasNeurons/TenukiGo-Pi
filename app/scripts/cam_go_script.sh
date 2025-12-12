#!/bin/bash

OUTPUT_DIR="/app/go_videos"
SGF_OUTPUT_DIR="/app/output_sgf"
MODELS_DIR="/app/models"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SGF_OUTPUT_DIR"

# Nommage des fichiers
START_TS=$(date +"%Y-%m-%d_%H-%M-%S")
SESSION_PREFIX="tmp_${START_TS}_part_"
FINAL_NAME="go_${START_TS}.h264"
FINAL_SGF="go_${START_TS}.sgf"

echo "=== Début Enregistrement ==="
echo "Fichier : $FINAL_NAME"

finalize() {
  echo ""
  echo "=== Arrêt détecté. Finalisation... ==="
  cd "$OUTPUT_DIR" || exit 1

  if ls ${SESSION_PREFIX}*.h264 1>/dev/null 2>&1; then
    echo "Concaténation des segments..."
    cat ${SESSION_PREFIX}*.h264 > "$FINAL_NAME"
    rm ${SESSION_PREFIX}*.h264
    echo "Vidéo assemblée : $OUTPUT_DIR/$FINAL_NAME"
  else
    echo "Aucun segment trouvé."
    exit 1
  fi

  echo "Lancement de l'analyse SGF..."
  
  python3 /app/main.py \
    --video "$OUTPUT_DIR/$FINAL_NAME" \
    --output "$SGF_OUTPUT_DIR/$FINAL_SGF" \
    --yolo-model "$MODELS_DIR/model.pt" \
    --keras-model "$MODELS_DIR/modelCNN.tflite" \
    --transparent # Mode différé

  echo "SGF généré : $SGF_OUTPUT_DIR/$FINAL_SGF"
  exit 0
}

# Intercepte Ctrl+C (SIGINT) et l'arrêt Docker (SIGTERM)
trap finalize INT TERM

# Commande d'enregistrement (Boucle infinie)
# Utilise rpicam-vid et ffmpeg pour couper en segments (sécurité en cas de crash)
rpicam-vid --width 1280 --height 720 --framerate 25 --inline -t 0 -o - | \
ffmpeg -re -f h264 -i - -c:v copy -f segment -segment_time 10 \
       -reset_timestamps 1 -segment_format h264 \
       "${OUTPUT_DIR}/${SESSION_PREFIX}%03d.h264"

# Au cas où la boucle casse toute seule
finalize