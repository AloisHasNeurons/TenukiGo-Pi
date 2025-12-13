#!/usr/bin/env python3
import http.server
import socketserver
import urllib.parse
import subprocess
import os

# Configuration
PORT = 80

# 1. On récupère le dossier où se trouve ce script (ex: /home/mao/scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. On remonte au dossier parent (ex: /home/mao)
home_dir = os.path.dirname(script_dir)

# 3. On vise le dossier des vidéos
target_dir = os.path.join(home_dir, "go_videos")

try:
    os.chdir(target_dir)
    print(f"Dossier de travail défini sur : {target_dir}")
except FileNotFoundError:
    print(f"⚠️ Dossier {target_dir} introuvable. Utilisation du dossier courant.")

VIDEO_DIR = os.getcwd()
WPA_FILE = "/etc/wpa_supplicant/wpa_supplicant.conf"


class WifiHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            # 1. Lister les vidéos
            files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.h264', '.mp4', '.mkv'))]
            files_html = "".join([f'<li><a href="{f}">{f}</a></li>' for f in files])

            # 2. Construire la page HTML
            html = f"""
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: sans-serif; padding: 20px; max-width: 600px; margin: auto; }}
                    h2 {{ border-bottom: 2px solid #333; }}
                    input {{ width: 100%; padding: 10px; margin: 5px 0; }}
                    button {{ background: #007bff; color: white; padding: 10px 20px; border: none; width: 100%; font-size: 16px; }}
                    .video-list {{ background: #f0f0f0; padding: 15px; border-radius: 8px; }}
                    li {{ margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>TenukiGo Setup</h1>
                
                <div class="video-list">
                    <h2>Mes Vidéos</h2>
                    <ul>{files_html if files else "<li>Aucune vidéo trouvée</li>"}</ul>
                </div>

                <div style="margin-top: 40px;">
                    <h2>Configurer le WiFi</h2>
                    <form method="POST">
                        <label>Nom du réseau (SSID)</label>
                        <input type="text" name="ssid" placeholder="Ex: Livebox-1234" required>
                        
                        <label>Mot de passe</label>
                        <input type="password" name="password" placeholder="Clé de sécurité" required>
                        
                        <button type="submit">Enregistrer et Redémarrer</button>
                    </form>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            # Keep default behaviour
            super().do_GET()

    def do_POST(self):
        # Récupération des données du formulaire
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = urllib.parse.parse_qs(post_data)

        ssid = params.get('ssid', [''])[0]
        password = params.get('password', [''])[0]

        if ssid and password:
            # Création de la config WPA
            config_content = (
                f"""country=FR
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""
            )
            try:
                # Écriture dans le fichier système
                with open(WPA_FILE, "w") as f:
                    f.write(config_content)
                # Réponse à l'utilisateur
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(b"<h1>Sauvegarde OK !</h1><p>La Raspberry Pi redemarre...</p>")
                # On lance le reboot
                subprocess.Popen(["reboot"])
            except Exception as e:
                self.send_error(500, f"Erreur d'ecriture : {e}")
        else:
            self.send_error(400, "Champs manquants")


# Démarrage du serveur
with socketserver.TCPServer(("", PORT), WifiHandler) as httpd:
    print(f"Serveur WiFi actif sur le port {PORT}")
    print(f"Dossier servi : {VIDEO_DIR}")
    httpd.serve_forever()
