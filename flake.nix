{
  description = "A reproducible Python environment with uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # --- NEW: Define Python version ONCE ---
        python = pkgs.python311;

        # Load the workspace from uv.lock
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # Create overlay from workspace
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Helper to patch NVIDIA packages
        patchNvidiaPackage = name: prev:
          if prev ? ${name} then {
            ${name} = prev.${name}.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;
            });
          } else {};

        # Create a Python package set with pyproject-nix and uv2nix
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python; # <-- USE THE 3.12 PYTHON
          }).overrideScope
            (pkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              # Force everything to use opencv-python-headless
              (final: prev: {
                "opencv-python" = final."opencv-python-headless";
              })
              # Override to skip broken CUDA/Torch packages
              (final: prev:
                pkgs.lib.foldl' (acc: name: acc // (patchNvidiaPackage name prev)) {} [
                  "torch"
                  "torchvision"
                  "nvidia-cufile-cu12"
                  "nvidia-nvshmem-cu12"
                  "nvidia-cuda-runtime-cu12"
                  "nvidia-cuda-nvrtc-cu12"
                  "nvidia-cublas-cu12"
                  "nvidia-cufft-cu12"
                  "nvidia-curand-cu12"
                  "nvidia-cusolver-cu12"
                  "nvidia-cusparse-cu12"
                  "nvidia-cudnn-cu12"
                  "nvidia-nvjitlink-cu12"
                  "nvidia-nvtx-cu12"
                ]
              )
            ]);

        # Get the list of Nix-built Python package derivations
        nixBuiltPythonDeps = pythonSet.resolveVirtualEnv workspace.deps.default;

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv

            # --- MODIFIED: Use the 3.12 python and its packages ---
            python # This is pkgs.python312
            python.pkgs.pip # This is the pip for 3.12
            pkgs.meson
            pkgs.ninja
            pkgs.gcc
            pkgs.git
          ]
          # Add all the Nix-built Python packages to the shell
          ++ nixBuiltPythonDeps;

          # This hook now creates a local venv
          shellHook = ''
            VENV_DIR=".venv"
            SENTE_SRC_DIR=".venv/sente-src"

            # Check for a broken venv
            if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
              echo "---"
              echo "Broken venv detected (missing pip). Removing $VENV_DIR..."
              rm -rf "$VENV_DIR"
              echo "Done."
              echo "---"
            fi

            # Create the venv if it doesn't exist
            if [ ! -d "$VENV_DIR" ]; then
              echo "---"
              echo "Creating new venv in $VENV_DIR..."
              ${python.interpreter} -m venv $VENV_DIR --system-site-packages
              echo "Done."
              echo "---"
            fi

            # Activate the venv
            source "$VENV_DIR/bin/activate"

            echo "---"
            echo "Cloning and patching 'sente' source..."
            # Clone the repo if it doesn't exist
            if [ ! -d "$SENTE_SRC_DIR" ]; then
              git clone https://github.com/atw1020/sente.git "$SENTE_SRC_DIR"
            else
              echo "Source directory already exists, skipping clone."
            fi

            # Patch 1: Fix broken setup.py metadata
            sed -i 's/">=3.8.*"/">=3.8"/' "$SENTE_SRC_DIR/setup.py"
            echo "Patched setup.py."

            # --- NEW PATCHES ---
            # Patch 2: Fix C++ build error by adding missing #include <algorithm>
            sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/Tree.h"
            sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/SGF/SGFNode.cpp"
            echo "Patched C++ source files."
            # -------------------

            echo "Installing 'sente' from patched local source..."
            # Install from our locally cloned and patched directory
            "$VENV_DIR/bin/pip" install --no-cache-dir --no-deps "$SENTE_SRC_DIR"
            echo "Done."
            echo "---"
          '';
        };
      });
}