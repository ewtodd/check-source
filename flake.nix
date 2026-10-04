{
  description = "Check Source: reference benchmarks for OpenAI-compatible LLM endpoints";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # math-verify (the MATH-500 Minerva scorer) is not in nixpkgs, and lm-eval
      # pins antlr4-python3-runtime==4.11 while nixpkgs ships 4.13.2. The overlay
      # adds the exact versions used by lm-evaluation-harness 0.4.13.
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ (import ./nix/math-verify-overlay.nix) ];
        };

      mathDeps = pkgs: [
        pkgs.python3Packages.datasets
        pkgs.python3Packages.sympy
        pkgs.python3Packages.math-verify
      ];

      checkSource =
        pkgs:
        pkgs.python3Packages.buildPythonApplication {
          pname = "check-source";
          version = "0.1.0";
          pyproject = true;
          src = ./.;
          build-system = [ pkgs.python3Packages.setuptools ];
          dependencies = mathDeps pkgs;
          # The benches talk to a live endpoint; there is nothing to unit test.
          doCheck = false;
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = checkSource pkgs;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShell {
            packages = [ (pkgs.python3.withPackages (mathDeps pkgs)) ];
          };
        }
      );

      formatter = forAllSystems (system: (pkgsFor system).nixfmt-rfc-style);
    };
}
