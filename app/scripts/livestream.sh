#!/bin/bash

# A remplacer avec l'adresse du serveur
RTMP_URL="rtmp://a.rtmp.youtube.com/live2/VOTRE_CLE_ICI"

rpicam-vid --width 1280 --height 720 --framerate 30 -o - --inline -n | \
ffmpeg -i - -f lavfi -i anullsrc \
  -c:v copy -b:v 3000k -bsf:v h264_mp4toannexb \
  -c:a aac -b:a 128k -ar 44100 \
  -f flv "$RTMP_URL"