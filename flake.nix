{
  description = "GSM8K 5-shot evaluator for OpenAI-compatible completion endpoints";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pythonFor =
        pkgs:
        pkgs.python3.withPackages (ps: [
          ps.datasets
        ]);
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.writeShellApplication {
            name = "gsm8k-eval";
            runtimeInputs = [ (pythonFor pkgs) ];
            text = ''
              exec python ${./gsm8k_eval.py} "$@"
            '';
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [ (pythonFor pkgs) ];
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);
    };
}
