#!/bin/bash
# Discovers the Raspberry Pi IP address via mDNS and updates the Ansible inventory.
#
# Usage:
#   ./find_pi.sh [hostname]
#
# Arguments:
#   hostname: The hostname of the Raspberry Pi (default: tenukigo-pi)

set -u

readonly DEFAULT_HOSTNAME="tenukigo-pi"

# Resolve absolute path to the project root based on script location.
# This ensures the script works regardless of the current working directory.
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly INVENTORY_FILE="${PROJECT_ROOT}/ansible/inventory.ini"

resolve_ip() {
  local hostname_local="$1"
  local ip_address=""

  # Try standard system resolver first.
  if command -v getent >/dev/null 2>&1; then
    ip_address=$(getent hosts "$hostname_local" | awk '{ print $1 }' | head -n 1)
  fi

  # Fallback to parsing ping output if getent failed or returned nothing.
  # This is often necessary on macOS or systems with distinct mDNS resolvers.
  if [[ -z "$ip_address" ]]; then
    ip_address=$(ping -c 1 "$hostname_local" 2>/dev/null \
      | sed -n 's/.*(\([0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\)).*/\1/p' \
      | head -n 1)
  fi

  echo "$ip_address"
}

main() {
  local hostname="${1:-$DEFAULT_HOSTNAME}"
  local hostname_local="${hostname}.local"

  local ip_address
  ip_address=$(resolve_ip "$hostname_local")

  if [[ -z "$ip_address" ]]; then
    echo "Error: Could not resolve '${hostname_local}'. Verify the device is powered on." >&2
    exit 1
  fi

  if [[ ! -f "$INVENTORY_FILE" ]]; then
    echo "Error: Inventory file not found at ${INVENTORY_FILE}" >&2
    exit 1
  fi

  # Update the IP address and user in the inventory file.
  # Matches lines starting with an IP and replaces both the IP and the ansible_user (if present).
  sed -i -E "s/^(\s*)[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(\s+ansible_user=)\S+/\1${ip_address}\2${hostname}/" "$INVENTORY_FILE"

  echo "Successfully updated inventory for ${hostname} (${ip_address})."
}

main "$@"
