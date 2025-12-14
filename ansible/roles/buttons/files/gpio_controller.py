#!/usr/bin/env python3
"""
Script de contrôle GPIO pour TenukiGo
Gère les boutons et LEDs pour capture vidéo, streaming et WiFi
Auteur : Équipe TenukiGo
Date : Décembre 2025
"""

import subprocess
import time
import os
import signal
from gpiozero import LED, Button
from signal import pause

# ============================================================================
# CONFIGURATION DES GPIO
# ============================================================================

# LED RGB (WiFi)
LED_RGB_VERTE = LED(17)     # Mode client WiFi (connecté à un réseau)
LED_RGB_BLEUE = LED(27)     # Mode AP WiFi (point d'accès)
BOUTON_RGB = Button(22, pull_up=True, bounce_time=0.3)

# LED simples (Capture/Streaming)
LED_VERTE_SIMPLE = LED(5)   # Capture/Streaming actif
LED_ROUGE_SIMPLE = LED(6)   # Erreur/Arrêt
BOUTON_VERT = Button(23, pull_up=True, bounce_time=0.3)
BOUTON_ROUGE = Button(24, pull_up=True, bounce_time=0.3)

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

# Processus en cours
capture_process = None
streaming_process = None

# États du système
is_capturing = False
is_streaming = False
is_ap_mode = False

# Chemins des scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_SCRIPT = f"{SCRIPT_DIR}/record_game.sh"
STREAMING_SCRIPT = f"{SCRIPT_DIR}/stream_video.sh"
AP_SCRIPT = f"{SCRIPT_DIR}/ap_mode_enable.sh"
CLIENT_SCRIPT = f"{SCRIPT_DIR}/client_mode_enable.sh"

# ============================================================================
# FONCTIONS DE GESTION DES LEDs
# ============================================================================

def update_wifi_leds():
    """Met à jour les LEDs RGB selon le mode WiFi"""
    if is_ap_mode:
        LED_RGB_VERTE.off()
        LED_RGB_BLEUE.on()
        print("📡 LEDs WiFi : Mode AP (Bleue ON)")
    else:
        LED_RGB_VERTE.on()
        LED_RGB_BLEUE.off()
        print("📶 LEDs WiFi : Mode Client (Verte ON)")

def blink_led(led, times=3, duration=0.2):
    """Fait clignoter une LED un certain nombre de fois"""
    for _ in range(times):
        led.on()
        time.sleep(duration)
        led.off()
        time.sleep(duration)

def show_error():
    """Affiche une erreur avec la LED rouge clignotante"""
    print("❌ ERREUR détectée")
    for _ in range(5):
        LED_ROUGE_SIMPLE.on()
        time.sleep(0.1)
        LED_ROUGE_SIMPLE.off()
        time.sleep(0.1)

def show_success():
    """Affiche un succès avec la LED verte clignotante"""
    print("✅ Opération réussie")
    blink_led(LED_VERTE_SIMPLE, times=2, duration=0.3)

# ============================================================================
# FONCTIONS DE GESTION CAPTURE/STREAMING
# ============================================================================

def start_capture():
    """Démarre l'enregistrement vidéo local"""
    global capture_process, is_capturing
    
    if is_capturing:
        print("⚠️ Capture déjà en cours")
        return
    
    print("🎥 Démarrage de la capture vidéo...")
    
    try:
        # Lancer le script de capture en arrière-plan
        capture_process = subprocess.Popen(
            [CAPTURE_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Crée un nouveau groupe de processus
        )
        
        # Attendre 2 secondes pour vérifier que ça démarre bien
        time.sleep(2)
        
        if capture_process.poll() is None:  # Processus toujours actif
            is_capturing = True
            LED_VERTE_SIMPLE.on()
            LED_ROUGE_SIMPLE.off()
            print("✅ Capture démarrée (PID: {})".format(capture_process.pid))
            show_success()
        else:
            print("❌ Échec démarrage capture")
            show_error()
            capture_process = None
            
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        show_error()
        capture_process = None

def start_streaming():
    """Démarre le streaming live YouTube"""
    global streaming_process, is_streaming
    
    if is_streaming:
        print("⚠️ Streaming déjà en cours")
        return
    
    print("📡 Démarrage du streaming YouTube Live...")
    
    try:
        # Lancer le script de streaming en arrière-plan
        streaming_process = subprocess.Popen(
            [STREAMING_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        time.sleep(2)
        
        if streaming_process.poll() is None:
            is_streaming = True
            LED_VERTE_SIMPLE.on()
            LED_ROUGE_SIMPLE.off()
            print("✅ Streaming démarré (PID: {})".format(streaming_process.pid))
            show_success()
        else:
            print("❌ Échec démarrage streaming")
            show_error()
            streaming_process = None
            
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        show_error()
        streaming_process = None

def stop_capture():
    """Arrête l'enregistrement vidéo"""
    global capture_process, is_capturing
    
    if not is_capturing or capture_process is None:
        print("⚠️ Aucune capture en cours")
        return
    
    print("⏹️ Arrêt de la capture...")
    LED_VERTE_SIMPLE.blink(on_time=0.2, off_time=0.2, n=5, background=False)
    
    try:
        # Envoyer SIGINT (équivalent Ctrl+C) pour terminaison propre
        os.killpg(os.getpgid(capture_process.pid), signal.SIGINT)
        
        # Attendre la fin du processus (max 30 secondes pour finalisation)
        try:
            capture_process.wait(timeout=30)
            print("✅ Capture arrêtée proprement")
        except subprocess.TimeoutExpired:
            print("⚠️ Timeout, arrêt forcé")
            os.killpg(os.getpgid(capture_process.pid), signal.SIGKILL)
        
        is_capturing = False
        capture_process = None
        LED_VERTE_SIMPLE.off()
        LED_ROUGE_SIMPLE.on()
        time.sleep(1)
        LED_ROUGE_SIMPLE.off()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'arrêt : {e}")
        show_error()

def stop_streaming():
    """Arrête le streaming live"""
    global streaming_process, is_streaming
    
    if not is_streaming or streaming_process is None:
        print("⚠️ Aucun streaming en cours")
        return
    
    print("⏹️ Arrêt du streaming...")
    LED_VERTE_SIMPLE.blink(on_time=0.2, off_time=0.2, n=5, background=False)
    
    try:
        os.killpg(os.getpgid(streaming_process.pid), signal.SIGINT)
        
        try:
            streaming_process.wait(timeout=10)
            print("✅ Streaming arrêté proprement")
        except subprocess.TimeoutExpired:
            print("⚠️ Timeout, arrêt forcé")
            os.killpg(os.getpgid(streaming_process.pid), signal.SIGKILL)
        
        is_streaming = False
        streaming_process = None
        LED_VERTE_SIMPLE.off()
        LED_ROUGE_SIMPLE.on()
        time.sleep(1)
        LED_ROUGE_SIMPLE.off()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'arrêt : {e}")
        show_error()

# ============================================================================
# FONCTIONS DE GESTION WIFI
# ============================================================================

def toggle_wifi_mode():
    """Bascule entre mode AP et mode Client"""
    global is_ap_mode
    
    print("🔄 Basculement mode WiFi...")
    
    # Clignotement pour indiquer le changement
    blink_led(LED_RGB_VERTE, times=3, duration=0.2)
    blink_led(LED_RGB_BLEUE, times=3, duration=0.2)
    
    try:
        if is_ap_mode:
            # Passage en mode Client
            print("📶 Passage en mode Client...")
            result = subprocess.run(
                ["sudo", CLIENT_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                is_ap_mode = False
                update_wifi_leds()
                print("✅ Mode Client activé")
                show_success()
            else:
                print(f"❌ Échec passage mode Client : {result.stderr}")
                show_error()
        else:
            # Passage en mode AP
            print("📡 Passage en mode AP...")
            result = subprocess.run(
                ["sudo", AP_SCRIPT],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                is_ap_mode = True
                update_wifi_leds()
                print("✅ Mode AP activé")
                show_success()
            else:
                print(f"❌ Échec passage mode AP : {result.stderr}")
                show_error()
                
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors du changement de mode WiFi")
        show_error()
    except Exception as e:
        print(f"❌ Erreur changement mode WiFi : {e}")
        show_error()

# ============================================================================
# GESTIONNAIRES D'ÉVÉNEMENTS BOUTONS
# ============================================================================

def on_bouton_vert_pressed():
    """Gestion appui bouton vert (Démarrage capture ET diffusion simultanées)"""
    print("\n🟢 Bouton VERT appuyé - Démarrage capture + diffusion")
    
    if is_capturing or is_streaming:
        print("⚠️ Déjà en cours d'enregistrement/diffusion")
        blink_led(LED_VERTE_SIMPLE, times=2, duration=0.1)
        return
    
    # Démarrer capture ET streaming en parallèle
    print("🎥 Lancement capture + diffusion simultanées...")
    
    # Démarrer la capture
    start_capture()
    time.sleep(1)  # Attendre que la capture démarre
    
    # Démarrer le streaming
    start_streaming()
    
    # Si les deux ont démarré avec succès
    if is_capturing and is_streaming:
        print("✅ Capture + Diffusion actives")
        LED_VERTE_SIMPLE.on()
        LED_ROUGE_SIMPLE.off()
    else:
        print("⚠️ Problème au démarrage")
        # Arrêter ce qui a pu démarrer
        if is_capturing:
            stop_capture()
        if is_streaming:
            stop_streaming()
        show_error()

def on_bouton_rouge_pressed():
    """Gestion appui bouton rouge (Arrêt capture ET diffusion - comme Ctrl+C)"""
    print("\n🔴 Bouton ROUGE appuyé - Arrêt de tout")
    
    if not is_capturing and not is_streaming:
        print("⚠️ Rien à arrêter")
        blink_led(LED_ROUGE_SIMPLE, times=2, duration=0.1)
        return
    
    # Arrêter les deux en parallèle
    print("⏹️ Arrêt capture + diffusion...")
    
    # Clignotement pendant l'arrêt
    LED_VERTE_SIMPLE.blink(on_time=0.2, off_time=0.2, n=5, background=True)
    
    # Arrêter streaming d'abord
    if is_streaming:
        stop_streaming()
    
    # Puis arrêter capture
    if is_capturing:
        stop_capture()
    
    # Allumer LED rouge pour indiquer l'arrêt
    LED_VERTE_SIMPLE.off()
    LED_ROUGE_SIMPLE.on()
    time.sleep(1)
    LED_ROUGE_SIMPLE.off()
    
    print("✅ Tout est arrêté")

def on_bouton_rgb_pressed():
    """Gestion appui bouton RGB (Basculement WiFi)"""
    print("\n🔵 Bouton RGB appuyé")
    toggle_wifi_mode()

# ============================================================================
# INITIALISATION ET BOUCLE PRINCIPALE
# ============================================================================

def cleanup():
    """Nettoyage lors de l'arrêt du script"""
    print("\n🛑 Arrêt du contrôleur GPIO...")
    
    # Arrêter les processus en cours
    if is_capturing:
        stop_capture()
    if is_streaming:
        stop_streaming()
    
    # Éteindre toutes les LEDs
    LED_RGB_VERTE.off()
    LED_RGB_BLEUE.off()
    LED_VERTE_SIMPLE.off()
    LED_ROUGE_SIMPLE.off()
    
    print("✅ Nettoyage terminé")

def init_system():
    """Initialisation du système au démarrage"""
    print("=" * 60)
    print("  CONTRÔLEUR GPIO TENUKIGO")
    print("=" * 60)
    print("")
    print("Configuration des GPIO :")
    print("  LED RGB Verte  (GPIO 17) : Mode Client WiFi")
    print("  LED RGB Bleue  (GPIO 27) : Mode AP WiFi")
    print("  Bouton RGB     (GPIO 22) : Basculer mode WiFi (Client ↔ AP)")
    print("")
    print("  LED Verte      (GPIO 5)  : Capture + Diffusion actives")
    print("  LED Rouge      (GPIO 6)  : Arrêt (temporaire)")
    print("  Bouton Vert    (GPIO 23) : Démarrer capture + diffusion")
    print("  Bouton Rouge   (GPIO 24) : Arrêter capture + diffusion (Ctrl+C)")
    print("")
    
    # Détecter le mode WiFi actuel
    global is_ap_mode
    try:
        result = subprocess.run(
            ["iw", "dev", "wlan0", "info"],
            capture_output=True,
            text=True
        )
        if "type AP" in result.stdout:
            is_ap_mode = True
            print("🔵 Mode WiFi détecté : AP (Point d'accès)")
        else:
            is_ap_mode = False
            print("🟢 Mode WiFi détecté : Client")
    except:
        is_ap_mode = False
        print("⚠️ Impossible de détecter le mode WiFi, défaut : Client")
    
    # Initialiser les LEDs WiFi
    update_wifi_leds()
    
    # Éteindre les LEDs de capture
    LED_VERTE_SIMPLE.off()
    LED_ROUGE_SIMPLE.off()
    
    # Test des LEDs au démarrage
    print("")
    print("🔄 Test des LEDs...")
    for led in [LED_RGB_VERTE, LED_RGB_BLEUE, LED_VERTE_SIMPLE, LED_ROUGE_SIMPLE]:
        led.on()
        time.sleep(0.3)
        led.off()
    
    # Restaurer l'état WiFi
    update_wifi_leds()
    
    print("")
    print("✅ Système initialisé")
    print("=" * 60)
    print("En attente d'actions utilisateur...")
    print("")

def main():
    """Fonction principale"""
    # Initialisation
    init_system()
    
    # Enregistrement des gestionnaires de boutons
    BOUTON_VERT.when_pressed = on_bouton_vert_pressed
    # Plus besoin de when_held pour le bouton vert
    BOUTON_ROUGE.when_pressed = on_bouton_rouge_pressed
    BOUTON_RGB.when_pressed = on_bouton_rgb_pressed
    
    # Configuration des signaux pour nettoyage propre
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup() or exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup() or exit(0))
    
    try:
        # Boucle infinie (écoute des événements GPIO)
        print("👂 Écoute des boutons...")
        pause()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"❌ Erreur : {e}")
        cleanup()

if __name__ == "__main__":
    main()
