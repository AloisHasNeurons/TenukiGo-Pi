#!/bin/bash
# Script pour diffuser la vidéo de la Raspberry Pi vers YouTube Live

rpicam-vid --width 1280 --height 720 --framerate 30 -o - --inline -n | \
ffmpeg -i - -f lavfi -i anullsrc \
  -c:v copy -b:v 3000k -bsf:v h264_mp4toannexb \
  -c:a aac -b:a 128k -ar 44100 \
  -f flv "rtmp://a.rtmp.youtube.com/live2/6v7e-8h0e-3130-p287-esfu"
