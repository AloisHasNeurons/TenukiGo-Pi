#!/bin/bash
set -e

echo "== Passage en mode Client (v2) =="

# 1) Stop Custom Server processes strictly
# We use pkill -f to match the command line. 
# "python3 .../server_wifi.py" handling covers the custom server
sudo pkill -f "python3 -m http.server" 2>/dev/null || true
sudo pkill -f "server_wifi.py" 2>/dev/null || true

# 2) Stop AND Disable AP services
# Disable is CRITICAL to prevent them from starting on reboot and conflicting with NetworkManager
sudo systemctl stop hostapd 2>/dev/null || true
sudo systemctl disable hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true
sudo systemctl disable dnsmasq 2>/dev/null || true

# 3) Cleanup Interface
# Remove the static IP from AP mode
sudo ip addr flush dev wlan0

# 4) Start Network Manager
# NetworkManager will automatically manage wpa_supplicant.
# We do NOT start wpa_supplicant manually to avoid "device busy" or locking issues.
sudo systemctl start NetworkManager 2>/dev/null || true

echo "✅ Mode client actif (v2) : NetworkManager gère la connexion."
