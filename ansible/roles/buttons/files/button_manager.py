#!/usr/bin/env python3
# Système de contrôle TenukiGo (Adapté Docker)

import RPi.GPIO as GPIO
import subprocess
import time
import os
import signal
import sys

# --- CONFIGURATION GPIO (Vos pins) ---
LED_RGB_VERTE = 17
LED_RGB_BLEUE = 27
BOUTON_RGB = 22

LED_VERTE_SIMPLE = 5
LED_ROUGE_SIMPLE = 6
BOUTON_VERT = 23
BOUTON_ROUGE = 24

# --- CONFIGURATION SYSTEME ---
# On pointe vers le script unique géré par Ansible
SCRIPT_NET_MODE = "/usr/local/bin/net_mode.sh"


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup LEDs
    GPIO.setup(LED_RGB_VERTE, GPIO.OUT)
    GPIO.setup(LED_RGB_BLEUE, GPIO.OUT)
    GPIO.setup(LED_VERTE_SIMPLE, GPIO.OUT)
    GPIO.setup(LED_ROUGE_SIMPLE, GPIO.OUT)

    # Setup Boutons (Pull-up interne)
    GPIO.setup(BOUTON_RGB, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BOUTON_VERT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BOUTON_ROUGE, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def run_cmd(cmd_list):
    """Lance une commande système"""
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur commande {cmd_list}: {e}")


# --- FONCTIONS PILOTAGE DOCKER ---
def demarrer_streaming(log_file):
    print(">>> Démarrage Docker...")
    GPIO.output(LED_VERTE_SIMPLE, GPIO.HIGH)
    # On lance le script bash DANS le conteneur
    run_cmd(["docker", "exec", "-d", "tenukigo-app", "/app/scripts/cam_go_script.sh"])


def arreter_streaming(log_file):
    print(">>> Arrêt & Analyse...")
    GPIO.output(LED_VERTE_SIMPLE, GPIO.LOW)
    # On envoie le signal d'arrêt au script dans Docker
    run_cmd(["docker", "exec", "tenukigo-app", "pkill", "-SIGINT", "-f", "cam_go_script.sh"])

    # Feedback visuel pendant l'analyse (optionnel)
    GPIO.output(LED_RGB_BLEUE, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(LED_RGB_BLEUE, GPIO.LOW)


def basculer_mode_wifi():
    print(">>> Bascule WiFi")
    # Logique de bascule simplifiée
    if os.path.exists("/tmp/ap_mode_active"):
        run_cmd(["sudo", SCRIPT_NET_MODE, "client"])
        os.remove("/tmp/ap_mode_active")
        GPIO.output(LED_RGB_VERTE, GPIO.HIGH)  # Vert = Client
        GPIO.output(LED_RGB_BLEUE, GPIO.LOW)
    else:
        run_cmd(["sudo", SCRIPT_NET_MODE, "ap"])
        open("/tmp/ap_mode_active", "w").close()
        GPIO.output(LED_RGB_VERTE, GPIO.LOW)
        GPIO.output(LED_RGB_BLEUE, GPIO.HIGH)  # Bleu = AP


# --- BOUCLE PRINCIPALE (Simplifiée) ---
def main():
    setup_gpio()
    print("TenukiGo Controller Prêt.")
    
    # Init état WiFi (supposé client au boot)
    GPIO.output(LED_RGB_VERTE, GPIO.HIGH) 

    try:
        while True:
            # Bouton WiFi
            if not GPIO.input(BOUTON_RGB):
                basculer_mode_wifi()
                time.sleep(0.5) # Debounce
            
            # Bouton Start (Vert)
            if not GPIO.input(BOUTON_VERT):
                demarrer_streaming(None)
                time.sleep(0.5)

            # Bouton Stop (Rouge)
            if not GPIO.input(BOUTON_ROUGE):
                arreter_streaming(None)
                time.sleep(0.5)

            time.sleep(0.1)

    except KeyboardInterrupt:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
