#!/bin/bash
set -e

VIDEO_DIR="/home/mao/go_videos"

echo "== Passage en mode Point d'accès (AP) =="

# 1) Stopper le mode client Wi-Fi (NetworkManager / wpa_supplicant)
sudo systemctl stop NetworkManager 2>/dev/null || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true

# 2) IP fixe 10.0.0.1 sur wlan0
sudo ip link set wlan0 down
sudo ip addr flush dev wlan0
sudo ip addr add 10.0.0.1/24 dev wlan0
sudo ip link set wlan0 up

# 3) Lancer DHCP (dnsmasq) et le point d'accès (hostapd)
sudo systemctl enable dnsmasq 2>/dev/null || true
sudo systemctl enable hostapd 2>/dev/null || true
sudo systemctl restart dnsmasq
sudo systemctl restart hostapd

# 4) Lancer serveur HTTP sur le dossier vidéos
echo "== (Re)démarrage du serveur HTTP sur $VIDEO_DIR =="

sudo pkill -f "python3 -m http.server 80" 2>/dev/null || true

sudo -u mao nohup bash -c "cd \"$VIDEO_DIR\" && python3 -m http.server 80" \
  > /home/mao/go_http.log 2>&1 &

echo "✅ Mode AP actif."
echo "Wi-Fi : WiFi-Tenuki (SSID)"
echo "Mot de passe : Go123456"
echo "Adresse à ouvrir : http://10.0.0.1"
