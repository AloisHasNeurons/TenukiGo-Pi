import asyncio
import websockets
import json
import threading

# URL du serveur de l'autre groupe (FastAPI)
FASTAPI_WS_URL = "ws://127.0.0.1:8000/ws/camera_feed"
PASSWORD = "clubgo2025"

async def send_sgf_to_server(sgf_content):
    """
    Se connecte au serveur FastAPI via WebSocket et envoie le SGF.
    """
    print(f"Envoi du SGF au serveur à {FASTAPI_WS_URL}...")
    
    partie_data = {
        "password": PASSWORD,
        "blanc_nom": "Cho",
        "blanc_prenom": "Chikun",
        "noir_nom": "Go",
        "noir_prenom": "Seigen",
        "sgf_content": sgf_content
    }
    
    try:
        async with websockets.connect(FASTAPI_WS_URL) as websocket:
            await websocket.send(json.dumps(partie_data))
            response = await websocket.recv()
            print(f"Réponse du serveur: {response}")
            
    except Exception as e:
        print(f"ERREUR d'envoi WebSocket: {e}")

def run_websocket_in_thread(sgf_content):
    """
    Wrapper pour lancer la fonction async dans un thread séparé.
    """
    try:
        asyncio.run(send_sgf_to_server(sgf_content))
    except Exception as e:
        print(f"Erreur thread WebSocket: {e}")