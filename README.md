# TenukiGo-Pi (IoT & IaC Migration)

TenukiGo-Pi is a fully containerized IoT system for recording and analyzing Go (Weiqi/Baduk) games on a Raspberry Pi. It leverages Computer Vision (YOLO + CNN) to digitize real-world games into SGF format automatically.

![Architecture](https://img.shields.io/badge/Architecture-Ansible%20%2B%20Docker-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%203B%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Architecture

The project follows a strict separation of concerns:

1.  **Infrastructure (Host - Ansible)**:
    *   System configuration (OS hardening, Docker Engine).
    *   Network Management (NetworkManager-based switching between Client and AP modes).
    *   Hardware Interface (Python daemon managing GPIO buttons & LEDs).
    *   Captive Portal for easy Wi-Fi configuration.

2.  **Application (Container - Docker)**:
    *   Video Capture (`rpicam-vid` optimized).
    *   Game Logic (`sente` library).
    *   Computer Vision (YOLOv8 for board detection + TensorFlow Lite for stone classification).
    *   Encapsulated in a `tenukigo-app` image based on `python:3.10-slim`.

---

## 🛠 Hardware Requirements

*   **Raspberry Pi 3B+** (or newer).
*   **Raspberry Pi Camera Module 2** (or newer).
*   **Physical Interface**:
    *   3 Push Buttons: **Green** (Start/Stop), **Red** (Power), **Blue** (Wi-Fi/AP Toggle).
    *   Status LEDs (integrated or external).

---

## Installation & Deployment

### Prerequisites (Control Machine)
*   User with SSH access to the Raspberry Pi.
*   `ansible` installed locally.
*   `podman` or `docker` (optional, for rebuilding the application image).

### Deployment

We provide a unified deployment script to handle discovery and provisioning.

1.  **Rebuild the Application Image** (If code changes were made):
    ```bash
    cd app
    podman build --no-cache --platform linux/arm64 -t tenukigo-app:latest .
    cd ..
    ```

2.  **Deploy to Raspberry Pi**:
    Run the deployment wrapper. It automatically discovers the Pi on your network (via mDNS) and triggers the Ansible playbook.
    ```bash
    ./deploy.sh [hostname]
    ```
    *Example: `./deploy.sh tenukigo-pi`*

    *The script will prompt for the `BECOME` password (the `sudo` password for the remote user on the Pi).*

---

## Usage

### Physical Interface
*   **Green Button**: 
    *   *Press*: Start recording/analysis session.
    *   *Press again*: Stop recording and trigger SGF generation.
*   **Blue Button (Wi-Fi)**: 
    *   *Press*: Toggle between **Client Mode** (Green LED) and **Access Point Mode** (Blue LED).
    *   *AP Mode*: Connect to Wi-Fi `TenukiGo-Pi` (Password: `123456`). A captive portal will open to configure your home Wi-Fi credentials.
*   **Red Button**: Power off the device safely.

### Retrieving Games
Generated `.sgf` files are stored in `~/output_sgf` on the Pi.
You can retrieve them via SCP or view them via the built-in HTTP server in AP mode.

---

## Project Structure

```text
.
├── ansible/            # Infrastructure as Code
│   ├── inventory.ini   # Target host configuration
│   ├── playbook.yml    # Main provisioning playbook
│   └── roles/
│       ├── common/     # Basic OS setup
│       ├── network/    # NetworkManager, hostapd, dnsmasq templates
│       ├── camera/     # Hardware configuration & capture scripts
│       └── docker/     # Container lifecycle management
├── app/                # Application Source Code
│   ├── Dockerfile      # Application image definition
│   ├── main.py         # Analysis entry point
│   ├── src/            # Python package (CV logic)
│   └── models/         # ML Models (YOLO/TFLite)
├── tools/              # Helper utilities (discovery scripts)
└── deploy.sh           # Deployment wrapper
```