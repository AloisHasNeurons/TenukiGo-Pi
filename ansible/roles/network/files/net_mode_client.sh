#!/bin/bash
set -e

echo "== Passage en mode Client =="

# 1) Arrêter le serveur HTTP
sudo pkill -f "python3 -m http.server 80" 2>/dev/null || true

# 2) Arrêter AP (hostapd + dnsmasq)
sudo systemctl stop hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true

# 3) Nettoyer l'IP de wlan0
sudo ip addr flush dev wlan0

# 4) Relancer le Wi-Fi normal
sudo systemctl start NetworkManager 2>/dev/null || true
sudo systemctl start wpa_supplicant 2>/dev/null || true

echo "✅ Mode client actif : la Pi se comporte comme avant."
