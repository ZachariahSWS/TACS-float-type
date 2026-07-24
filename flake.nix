{
  description = "TACS development environment";

  inputs.dev.url = "path:/home/zach/nixos-config/devshells";

  outputs = { dev, ... }:
    let
      system = "x86_64-linux";

      project = dev.lib.${system}.mkProject {
        name = "tacs";
        extraShellHook = ''
          if [ -x "$NIX_DEV_ROOT/.cuda-shim/ptxas" ]; then
            export TRITON_PTXAS_PATH="$NIX_DEV_ROOT/.cuda-shim/ptxas"
          elif [ -x "$CUDA_HOME/bin/ptxas" ]; then
            export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
          fi

          if [ -x "$NIX_DEV_ROOT/.cuda-shim/ptxas-blackwell" ]; then
            export TRITON_PTXAS_BLACKWELL_PATH="$NIX_DEV_ROOT/.cuda-shim/ptxas-blackwell"
          elif [ -x "$CUDA_HOME/bin/ptxas" ]; then
            export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
          fi
        '';
      };
    in
    {
      devShells.${system}.default = project.devShell;
    };
}
