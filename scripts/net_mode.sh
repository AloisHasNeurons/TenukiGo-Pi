#!/bin/bash

MODE="$1"

if [ "$MODE" = "ap" ]; then
    echo "== Passage en mode Point d'accès (AP) =="

    # dhcpcd en mode AP (wlan0 = 10.0.0.1)
    sudo cp /etc/dhcpcd.conf.ap /etc/dhcpcd.conf

    # Config AP pour dnsmasq
    sudo cp /etc/dnsmasq.conf.ap /etc/dnsmasq.conf

    # Désactiver le Wi-Fi client (optionnel selon ta config)
    sudo systemctl stop wpa_supplicant 2>/dev/null || true
    sudo systemctl stop NetworkManager 2>/dev/null || true

    # Activer les services AP
    sudo systemctl enable dnsmasq
    sudo systemctl enable hostapd
    sudo systemctl restart dhcpcd
    sudo systemctl restart dnsmasq
    sudo systemctl restart hostapd

    echo "✅ Mode AP actif."
    echo "Wi-Fi : Go-Camera (SSID)"
    echo "Mot de passe : gocamera123"
    echo "Adresse de la Pi : http://10.0.0.1"

elif [ "$MODE" = "client" ]; then
    echo "== Passage en mode Client (connexion à une box Wi-Fi) =="

    # dhcpcd en mode client (pas d'IP fixe sur wlan0)
    sudo cp /etc/dhcpcd.conf.client /etc/dhcpcd.conf

    # Si tu veux récupérer une config dnsmasq client (sinon on le désactive juste)
    if [ -f /etc/dnsmasq.conf.client ]; then
        sudo cp /etc/dnsmasq.conf.client /etc/dnsmasq.conf
    else
        sudo rm -f /etc/dnsmasq.conf
    fi

    # Désactiver les services AP
    sudo systemctl stop hostapd
    sudo systemctl stop dnsmasq
    sudo systemctl disable hostapd
    sudo systemctl disable dnsmasq

    sudo systemctl restart dhcpcd

    # Réactiver le client Wi-Fi (si tu utilises wpa_supplicant ou NetworkManager)
    sudo systemctl start wpa_supplicant 2>/dev/null || true
    sudo systemctl start NetworkManager 2>/dev/null || true

    echo "✅ Mode client actif."
    echo "La Pi se comporte comme avant (connexion à une box, SSH via IP de la box)."

else
    echo "Usage: sudo ~/scripts/net_mode.sh ap|client"
    exit 1
fi
