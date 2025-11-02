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
        # This reads the lock file and creates a structured representation
        # of all dependencies with their exact versions
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # Create overlay from workspace
        # sourcePreference = "wheel" means we prefer pre-built wheels over source distributions
        # This is faster and more reliable for packages with complex build requirements
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Helper function to patch NVIDIA CUDA packages
        # These packages often have missing dependencies that don't affect runtime
        # autoPatchelfIgnoreMissingDeps = true prevents build failures from these
        patchNvidiaPackage = name: prev:
          if prev ? ${name} then {
            ${name} = prev.${name}.overrideAttrs (old: {
              autoPatchelfIgnoreMissingDeps = true;
            });
          } else {};

        # Create a Python package set with all dependencies from uv.lock
        # This combines multiple overlays to:
        # 1. Add standard build systems (setuptools, etc.)
        # 2. Add our workspace dependencies from uv.lock
        # 3. Replace opencv-python with opencv-python-headless (no GUI dependencies)
        # 4. Patch NVIDIA CUDA packages to ignore missing ELF dependencies
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (pkgs.lib.composeManyExtensions [
              # Add standard Python build systems
              pyproject-build-systems.overlays.default
              # Add dependencies from uv.lock
              overlay
              # Force opencv-python to use the headless variant
              # This avoids pulling in X11 and GUI libraries
              (final: prev: {
                "opencv-python" = final."opencv-python-headless";
              })
              # Patch all NVIDIA CUDA packages to ignore missing dependencies
              # This is necessary because NVIDIA packages often reference libraries
              # that aren't needed in practice but cause autoPatchelf to fail
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
        # This returns a list of Nix derivations for all packages in the workspace
        nixBuiltPythonDeps = pythonSet.resolveVirtualEnv workspace.deps.default;

      in
      {
        # Default development shell
        # Usage: nix develop
        devShells.default = pkgs.mkShell {
          packages = [
            # uv for managing Python dependencies
            pkgs.uv

            # Python interpreter and pip
            python
            python.pkgs.pip

            # Build tools required for compiling Python packages with C extensions
            pkgs.meson
            pkgs.ninja
            pkgs.gcc
            pkgs.git
          ]
          # Add all Python packages from uv.lock
          ++ nixBuiltPythonDeps;

          # Shell hook runs when entering the development environment
          # It creates a virtual environment with --system-site-packages to access
          # the Nix-built packages, then installs 'sente' from a patched local clone
          shellHook = ''
            VENV_DIR=".venv"
            SENTE_SRC_DIR=".venv/sente-src"

            # Detect and remove broken venvs (e.g., if pip is missing)
            # This can happen if the venv was created with a different Python version
            if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/pip" ]; then
              echo "---"
              echo "Broken venv detected (missing pip). Removing $VENV_DIR..."
              rm -rf "$VENV_DIR"
              echo "Done."
              echo "---"
            fi

            # Create virtual environment with --system-site-packages
            # This allows the venv to access Nix-built packages while still
            # supporting pip install for packages not in uv.lock
            if [ ! -d "$VENV_DIR" ]; then
              echo "---"
              echo "Creating new venv in $VENV_DIR..."
              ${python.interpreter} -m venv $VENV_DIR --system-site-packages
              echo "Done."
              echo "---"
            fi

            # Activate the virtual environment
            source "$VENV_DIR/bin/activate"

            echo "---"
            echo "Cloning and patching 'sente' source..."

            # Clone the sente repository if not already present
            # sente is a Python library for Go (the board game) analysis
            if [ ! -d "$SENTE_SRC_DIR" ]; then
              git clone https://github.com/atw1020/sente.git "$SENTE_SRC_DIR"
            else
              echo "Source directory already exists, skipping clone."
            fi

            # Patch 1: Fix setup.py to allow Python 3.8+
            # Original setup.py has a malformed version specifier
            sed -i 's/">=3.8.*"/">=3.8"/' "$SENTE_SRC_DIR/setup.py"
            echo "Patched setup.py."

            # Patch 2: Fix C++ compilation errors
            # The source files use std::max_element without including <algorithm>
            sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/Tree.h"
            sed -i '1s;^;#include <algorithm>\n;' "$SENTE_SRC_DIR/src/Utils/SGF/SGFNode.cpp"
            echo "Patched C++ source files."

            echo "Installing 'sente' from patched local source..."
            # Install using --no-deps because dependencies are already provided by Nix
            # --no-cache-dir ensures we always use our patched version
            "$VENV_DIR/bin/pip" install --no-cache-dir --no-deps "$SENTE_SRC_DIR"
            echo "Done."
            echo "---"
          '';
        };
      });
}