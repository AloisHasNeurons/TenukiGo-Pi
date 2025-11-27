#!/bin/bash

# Dossier où les vidéos seront stockées
OUTPUT_DIR="/home/mao/go_videos"
mkdir -p "$OUTPUT_DIR"

# Horodatage du début de la partie (sert à nommer la vidéo finale)
START_TS=$(date +"%Y-%m-%d_%H-%M-%S")
SESSION_PREFIX="tmp_${START_TS}_part_"
FINAL_NAME="go_${START_TS}.h264"

echo "=== Début de l'enregistrement de la partie de Go ==="
echo "Horodatage : $START_TS"
echo "Dossier de sortie : $OUTPUT_DIR"
echo "Appuie sur Ctrl+C à la fin de la partie pour terminer."

finalize() {
  echo
  echo "=== Arrêt détecté, concaténation et nettoyage... ==="
  cd "$OUTPUT_DIR" || exit 1

  # Vérifier s'il existe des segments pour cette session
  if ls ${SESSION_PREFIX}*.h264 1>/dev/null 2>&1; then
    echo "Concaténation des segments en : $FINAL_NAME"
    cat ${SESSION_PREFIX}*.h264 > "$FINAL_NAME"

    echo "Suppression des segments temporaires..."
    rm ${SESSION_PREFIX}*.h264

    echo "✅ Fichier final conservé : $OUTPUT_DIR/$FINAL_NAME"
  else
    echo "⚠️ Aucun segment trouvé pour cette session, rien à concaténer."
  fi

  exit 0
}

# Quand on reçoit Ctrl+C (SIGINT) ou un arrêt (SIGTERM), on appelle finalize()
trap finalize INT TERM

# Enregistrement continu en segments de 30 minutes (1800 secondes)
rpicam-vid \
  --width 1280 --height 720 \
  --framerate 25 \
  --inline \
  -t 0 \
  -o - | \
ffmpeg -re -f h264 -i - \
  -c:v copy \
  -f segment \
  -segment_time 10 \
  -reset_timestamps 1 \
  -segment_format h264 \
  "${OUTPUT_DIR}/${SESSION_PREFIX}%03d.h264"

# Si le pipeline se termine normalement (sans Ctrl+C), on appelle aussi finalize()
finalize
