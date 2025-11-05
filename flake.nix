{
  description = "A reproducible Python environment with uv2nix for developing TenukiGo-Pi";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # pyproject-nix provides tools for building Python projects using PEP 621 pyproject.toml
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # uv2nix converts uv.lock files into Nix derivations
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    # Provides build systems (setuptools, poetry-core, etc.) as Nix packages
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

        # Pin to Python 3.11 for consistency across the project
        python = pkgs.python311;

        # Load the workspace from uv.lock
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # Create overlay from workspace
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Helper function to patch NVIDIA CUDA packages
        patchNvidiaPackage = name: prev:
          if prev ? ${name} then {
            ${name} = prev.${name}.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;
            });
          } else {};

        # Create a Python package set with all dependencies from uv.lock
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (pkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              (final: prev: {
                "opencv-python" = final."opencv-python-headless";
              })
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

        # Resolve the virtual environment's dependencies
        nixBuiltPythonDeps = pythonSet.resolveVirtualEnv workspace.deps.default;

        # Common shell hook for sente patching
        commonSenteSetup = ''
          SENTE_SRC_DIR=".venv/sente-src"

          echo "---"
          echo "Cloning and patching 'sente' source..."

          if [ ! -d "$SENTE_SRC_DIR" ]; then
            git clone https://github.com/atw1020/sente.git "$SENTE_SRC_DIR"
          else
            echo "Source directory already exists, skipping clone."
          fi

          # Patch setup.py and C++ files
          sed -i 's/">=3.8.*"/">=3.8"/' "$SENTE_SRC_DIR/setup.py"
          sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/Tree.h"
          sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/SGF/SGFNode.cpp"
          echo "Patched setup.py and C++ source files."

          echo "Installing 'sente' from patched local source..."
          "$VENV_DIR/bin/pip" install --no-cache-dir --no-deps "$SENTE_SRC_DIR"
          echo "Done."
          echo "---"
        '';

      in
      {
        devShells = {
          # Development shell - Uses editable install from ./src
          # This is the default when you run: nix develop
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
              python.pkgs.pip
              pkgs.meson
              pkgs.ninja
              pkgs.gcc
              pkgs.git
            ]
            ++ nixBuiltPythonDeps;

            shellHook = ''
              VENV_DIR=".venv"

              # Detect and remove broken venvs
              if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
                echo "---"
                echo "Broken venv detected (missing pip). Removing $VENV_DIR..."
                rm -rf "$VENV_DIR"
                echo "Done."
                echo "---"
              fi

              # Create venv with system-site-packages
              if [ ! -d "$VENV_DIR" ]; then
                echo "---"
                echo "Creating new development venv in $VENV_DIR..."
                ${python.interpreter} -m venv $VENV_DIR --system-site-packages
                echo "Done."
                echo "---"
              fi

              source "$VENV_DIR/bin/activate"

              # Install project in editable mode
              echo "---"
              echo "Installing tenukigo_pi in editable mode..."
              pip install -e . --no-deps
              echo "Done."
              echo "---"

              ${commonSenteSetup}

              echo "Development shell activated"
              echo "   - tenukigo_pi: editable install from ./src"
              echo "   - Dependencies: from Nix store"
              echo "   - Changes to src/ are immediately available"
            '';
          };

          # Production/deployment shell - Uses frozen Nix packages
          # Use with: nix develop .#prod
          prod = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
              python.pkgs.pip
              pkgs.meson
              pkgs.ninja
              pkgs.gcc
              pkgs.git
            ]
            ++ nixBuiltPythonDeps;

            shellHook = ''
              VENV_DIR=".venv"

              if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
                echo "---"
                echo "Broken venv detected (missing pip). Removing $VENV_DIR..."
                rm -rf "$VENV_DIR"
                echo "Done."
                echo "---"
              fi

              if [ ! -d "$VENV_DIR" ]; then
                echo "---"
                echo "Creating new production venv in $VENV_DIR..."
                ${python.interpreter} -m venv $VENV_DIR --system-site-packages
                echo "Done."
                echo "---"
              fi

              source "$VENV_DIR/bin/activate"

              ${commonSenteSetup}

              echo "Production shell activated"
              echo "   - tenukigo_pi: from Nix store (frozen)"
              echo "   - Dependencies: from Nix store"
              echo "   - Reproducible deployment environment"
            '';
          };

          # Raspberry Pi optimized shell - Lightweight, no CUDA
          # Use with: nix develop .#rpi
          rpi = pkgs.mkShell {
            packages = [
              pkgs.uv
              python
              python.pkgs.pip
              pkgs.meson
              pkgs.ninja
              pkgs.gcc
              pkgs.git
            ]
            # Note: You'd need to create a separate pythonSetRPi without CUDA
            # For now, using same deps but with environment variables
            ++ nixBuiltPythonDeps;

            shellHook = ''
              VENV_DIR=".venv"

              if [ ! -d "$VENV_DIR" ]; then
                echo "---"
                echo "Creating new RPi venv in $VENV_DIR..."
                ${python.interpreter} -m venv $VENV_DIR --system-site-packages
                echo "Done."
                echo "---"
              fi

              source "$VENV_DIR/bin/activate"

              # Install project in editable mode
              echo "---"
              echo "Installing tenukigo_pi in editable mode..."
              pip install -e . --no-deps
              echo "Done."
              echo "---"

              ${commonSenteSetup}

              # Set environment variables for RPi optimization
              export OPENBLAS_NUM_THREADS=4
              export OMP_NUM_THREADS=4
              export MKL_NUM_THREADS=4

              # Disable CUDA
              export CUDA_VISIBLE_DEVICES=""

              echo "Raspberry Pi shell activated"
              echo "   - tenukigo_pi: editable install from ./src"
              echo "   - CPU-only mode (CUDA disabled)"
              echo "   - Thread limits set for RPi"
            '';
          };
        };
      });
}