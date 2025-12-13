#!/bin/bash
set -e

VIDEO_DIR="/home/mao/go_videos"

echo "== Passage en mode Point d'accès (AP) =="

sudo systemctl unmask hostapd
sudo systemctl unmask dnsmasq

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

# 4) Lancer le serveur HTTP "Intelligent" (Vidéo + Setup WiFi)
echo "== (Re)démarrage du serveur Custom =="

# On tue tout ce qui pourrait tourner (le simple ou le custom)
sudo pkill -f "python3 -m http.server" 2>/dev/null || true
sudo pkill -f "server_wifi.py" 2>/dev/null || true

# On lance le nouveau script Python EN TANT QUE ROOT
# (Important pour écrire dans /etc/wpa_supplicant.conf et utiliser le port 80)
sudo nohup python3 /home/mao/scripts/server_wifi.py > /home/mao/go_http.log 2>&1 &

echo "✅ Mode AP actif."
echo "Wi-Fi : Go-Camera (SSID)"
echo "Mot de passe : gocamera123"
echo "Adresse à ouvrir : http://10.0.0.1"