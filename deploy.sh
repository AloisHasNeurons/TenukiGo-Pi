#!/bin/bash
# Wrapper script to automatically discover the Pi IP and run Ansible deployment.
#
# Usage:
#   ./deploy.sh [hostname]
#
# Arguments:
#   hostname: Optional hostname for discovery (default: tenukigo-pi)

set -euo pipefail

readonly DEFAULT_HOSTNAME="tenukigo-pi"
readonly HOSTNAME="${1:-$DEFAULT_HOSTNAME}"
readonly INVENTORY_FILE="ansible/inventory.ini"
readonly PLAYBOOK_FILE="ansible/playbook.yml"

main() {
  echo "--- Step 1: Discovering Device ---"
  if ./tools/find_pi.sh "$HOSTNAME"; then
      echo "✅ Device found and inventory updated."
  else
      echo "⚠️ Discovery failed. Proceeding with existing inventory configuration..."
  fi

  echo ""
  echo "--- Step 2: running Ansible Playbook ---"
  echo "You will be prompted for the BECOME password (sudo password for the Pi)."
  
  # Check if inventory exists
  if [[ ! -f "$INVENTORY_FILE" ]]; then
      echo "Error: Inventory file '$INVENTORY_FILE' not found."
      exit 1
  fi

  # Run Ansible
  # -i: Inventory file
  # -K: Ask for become (sudo) password
  ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_FILE" -K
}

main "$@"
