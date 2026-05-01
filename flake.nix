{
  description = "Python devshell + sandboxed Claude Code (uv + PyTorch CUDA 12.8 + NVIDIA passthrough)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    python = pkgs.python312;
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python
        uv
        zlib
        glibc.bin

        nodejs_20

        gcc
        binutils
        pkg-config

        git
        ripgrep
        fd

        bubblewrap
        cacert
      ];
      shellHook = ''
        export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
        export NODE_EXTRA_CA_CERTS="$SSL_CERT_FILE"

        LD_PATHS=(
          "${pkgs.stdenv.cc.cc.lib}/lib"
        )

        if [ -d /run/opengl-driver/lib ]; then
          LD_PATHS+=(/run/opengl-driver/lib)
        fi
        if [ -d /run/opengl-driver-32/lib ]; then
          LD_PATHS+=(/run/opengl-driver-32/lib)
        fi

        export LD_LIBRARY_PATH="$(IFS=:; echo "''${LD_PATHS[*]}")''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

        export UV_CACHE_DIR="$PWD/.uv-cache"
        export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"

        CUDA_SHIM_DIR="$PWD/.cuda-shim"
        mkdir -p "$CUDA_SHIM_DIR"

        if [ -e /run/opengl-driver/lib/libcuda.so.1 ]; then
          ln -sf /run/opengl-driver/lib/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so.1"
          ln -sf /run/opengl-driver/lib/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so"
          export TRITON_LIBCUDA_PATH="$CUDA_SHIM_DIR"
        elif [ -e /run/opengl-driver-32/lib/libcuda.so.1 ]; then
          ln -sf /run/opengl-driver-32/lib/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so.1"
          ln -sf /run/opengl-driver-32/lib/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so"
          export TRITON_LIBCUDA_PATH="$CUDA_SHIM_DIR"
        fi

        export LD_LIBRARY_PATH="$CUDA_SHIM_DIR:$LD_LIBRARY_PATH"

        echo
        echo "Python + Claude dev shell ready."
        echo "TRITON_LIBCUDA_PATH=''${TRITON_LIBCUDA_PATH:-<unset>}"
        echo
        echo "Suggested project bootstrap:"
        echo "  uv python install 3.12"
        echo "  uv venv --python 3.12"
        echo "  source .venv/bin/activate"
        echo "  uv add torch torchvision torchaudio --index pytorch=https://download.pytorch.org/whl/cu128"
        echo "  npm init -y"
        echo "  npm i -D @anthropic-ai/claude-code"
        echo "  nix run .#claude-sandbox"
        echo
      '';
    };

    packages.${system}.claude-sandbox = pkgs.writeShellApplication {
      name = "claude-sandbox";

      runtimeInputs = with pkgs; [
        bubblewrap
        nodejs_20
        python
        uv
        git
        ripgrep
        fd
        cacert
        gcc
        binutils
        pkg-config
        coreutils
        gnugrep
        gnused
        findutils
        bash
        glibc.bin
      ];

      text = ''
        set -euo pipefail

        ROOT="$(pwd -P)"
        CLAUDE_HOME="$ROOT/.claude-home"
        UV_CACHE="$ROOT/.uv-cache"
        UV_PYTHON="$ROOT/.uv-python"
        VENV_DIR="$ROOT/.venv"

        if [ ! -f "$ROOT/package.json" ]; then
          echo "Missing package.json."
          echo "Run:"
          echo "  nix develop"
          echo "  npm init -y"
          echo "  npm i -D @anthropic-ai/claude-code"
          exit 1
        fi

        CLI_JS="$ROOT/node_modules/@anthropic-ai/claude-code/cli.js"
        if [ ! -f "$CLI_JS" ]; then
          echo "Missing $CLI_JS."
          echo "Run:"
          echo "  nix develop"
          echo "  npm i -D @anthropic-ai/claude-code"
          exit 1
        fi

        mkdir -p \
          "$CLAUDE_HOME" \
          "$CLAUDE_HOME/.config" \
          "$CLAUDE_HOME/.cache" \
          "$CLAUDE_HOME/.local/state" \
          "$UV_CACHE" \
          "$UV_PYTHON" \
          "$VENV_DIR"

        if [ ! -x "$VENV_DIR/bin/python" ]; then
          echo "Missing project virtualenv at $VENV_DIR."
          echo "Run:"
          echo "  nix develop"
          echo "  uv venv --python 3.12"
          echo "  source .venv/bin/activate"
          echo "  uv sync"
          exit 1
        fi

        args=()
        args+=(--unshare-all)
        args+=(--share-net)
        args+=(--die-with-parent)
        args+=(--new-session)

        args+=(--proc /proc)
        args+=(--dev /dev)
        args+=(--tmpfs /tmp)

        args+=(--dir /sbin)
        args+=(--ro-bind ${pkgs.glibc.bin}/bin/ldconfig /sbin/ldconfig)

        args+=(--ro-bind /nix /nix)

        if [ -f /etc/resolv.conf ]; then
          args+=(--ro-bind /etc/resolv.conf /etc/resolv.conf)
        fi
        if [ -f /etc/hosts ]; then
          args+=(--ro-bind /etc/hosts /etc/hosts)
        fi
        if [ -f /etc/services ]; then
          args+=(--ro-bind /etc/services /etc/services)
        fi

        args+=(--ro-bind ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt)
        args+=("--setenv" "SSL_CERT_FILE" "/etc/ssl/certs/ca-bundle.crt")
        args+=("--setenv" "NODE_EXTRA_CA_CERTS" "/etc/ssl/certs/ca-bundle.crt")

        args+=(--bind "$ROOT" /workspace)

        if [ -f "$ROOT/flake.nix" ]; then
          args+=(--ro-bind "$ROOT/flake.nix" /workspace/flake.nix)
        fi
        if [ -f "$ROOT/flake.lock" ]; then
          args+=(--ro-bind "$ROOT/flake.lock" /workspace/flake.lock)
        fi

        args+=("--setenv" "HOME" "/workspace/.claude-home")
        args+=("--setenv" "USER" "sandbox")
        args+=("--setenv" "LOGNAME" "sandbox")
        args+=("--setenv" "XDG_CONFIG_HOME" "/workspace/.claude-home/.config")
        args+=("--setenv" "XDG_CACHE_HOME" "/workspace/.claude-home/.cache")
        args+=("--setenv" "XDG_STATE_HOME" "/workspace/.claude-home/.local/state")

        args+=("--setenv" "UV_CACHE_DIR" "/workspace/.uv-cache")
        args+=("--setenv" "UV_PYTHON_INSTALL_DIR" "/workspace/.uv-python")
        args+=("--setenv" "VIRTUAL_ENV" "/workspace/.venv")
        args+=("--setenv" "PYTHONNOUSERSITE" "1")

        for dev in \
          /dev/nvidiactl \
          /dev/nvidia-uvm \
          /dev/nvidia-uvm-tools \
          /dev/nvidia-modeset
        do
          if [ -e "$dev" ]; then
            args+=(--dev-bind "$dev" "$dev")
          fi
        done

        for dev in /dev/nvidia[0-9]*; do
          if [ -e "$dev" ]; then
            args+=(--dev-bind "$dev" "$dev")
          fi
        done

        if [ -d /dev/dri ]; then
          args+=(--dev-bind /dev/dri /dev/dri)
        fi

        if [ -d /run/opengl-driver ]; then
          args+=(--ro-bind /run/opengl-driver /run/opengl-driver)
        fi
        if [ -d /run/opengl-driver-32 ]; then
          args+=(--ro-bind /run/opengl-driver-32 /run/opengl-driver-32)
        fi

        if [ -d /sys ]; then
          args+=(--ro-bind /sys /sys)
        fi

        LD_PATHS=(
          "${pkgs.stdenv.cc.cc.lib}/lib"
        )
        if [ -d /run/opengl-driver/lib ]; then
          LD_PATHS+=(/run/opengl-driver/lib)
        fi
        if [ -d /run/opengl-driver-32/lib ]; then
          LD_PATHS+=(/run/opengl-driver-32/lib)
        fi
        args+=("--setenv" "LD_LIBRARY_PATH" "$(IFS=:; echo "''${LD_PATHS[*]}")")

        args+=("--setenv" "CUDA_DEVICE_ORDER" "PCI_BUS_ID")
        args+=("--setenv" "NVIDIA_VISIBLE_DEVICES" "all")
        args+=("--setenv" "NVIDIA_DRIVER_CAPABILITIES" "compute,utility")

        SANDBOX_PATH="/workspace/.venv/bin:${pkgs.lib.makeBinPath (with pkgs; [
          nodejs_20
          python
          uv
          git
          ripgrep
          fd
          gcc
          binutils
          pkg-config
          coreutils
          gnugrep
          gnused
          findutils
          bash
          glibc.bin
        ])}"

        args+=("--setenv" "PATH" "$SANDBOX_PATH")
        args+=(--chdir /workspace)

        exec ${pkgs.bubblewrap}/bin/bwrap "''${args[@]}" \
          ${pkgs.nodejs_20}/bin/node \
          /workspace/node_modules/@anthropic-ai/claude-code/cli.js \
          "$@"
      '';
    };
  };
}
