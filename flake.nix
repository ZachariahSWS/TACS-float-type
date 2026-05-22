{
  description = "TACS dev shell";

  inputs.dev.url = "path:/home/zach/nix-devshells";

  outputs = { self, dev }:
  let
    system = "x86_64-linux";

    env = dev.lib.${system}.mkProject {
      name = "tacs";
      claude = true;
    };
  in
  {
    devShells.${system}.default = env.devShell;
    packages.${system} = env.packages;
  };
}
