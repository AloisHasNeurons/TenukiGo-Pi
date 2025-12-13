#!/bin/bash
set -e

VIDEO_DIR="/home/mao/go_videos"

echo "== Passage en mode Point d'accès (AP) (v2) =="

# 0) Prepare services
sudo systemctl unmask hostapd
sudo systemctl unmask dnsmasq

# 1) Stop Client Mode Services
# We stop NetworkManager to prevent it from interfering with wlan0
sudo systemctl stop NetworkManager 2>/dev/null || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true

# Ensure wpa_supplicant is really dead so it doesn't hold the interface
sudo killall wpa_supplicant 2>/dev/null || true

# 2) Configure Static IP for AP
sudo ip link set wlan0 down
sudo ip addr flush dev wlan0
sudo ip addr add 10.0.0.1/24 dev wlan0
sudo ip link set wlan0 up

# 3) Start DHCP and AP services
# We ENABLE them so they persist if we reboot in AP mode (though client mode script must disable them)
sudo systemctl enable dnsmasq 2>/dev/null || true
sudo systemctl enable hostapd 2>/dev/null || true
sudo systemctl restart dnsmasq
sudo systemctl restart hostapd

# 4) Start Custom Web Server
echo "== (Re)démarrage du serveur Custom =="

# Kill any existing instances to avoid port conflicts
sudo pkill -f "python3 -m http.server" 2>/dev/null || true
sudo pkill -f "server_wifi.py" 2>/dev/null || true

# Launch the server
# Using full path to python3 and script for robustness
sudo nohup python3 /home/mao/scripts/server_wifi.py > /home/mao/go_http.log 2>&1 &

echo "✅ Mode AP actif (v2)."
echo "Wi-Fi : Go-Camera (SSID)"
echo "Mot de passe : gocamera123"
echo "Adresse à ouvrir : http://10.0.0.1"
