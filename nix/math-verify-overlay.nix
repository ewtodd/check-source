# Three pure-Python wheels that nixpkgs does not carry (or carries at a
# different version). They are pinned to exactly what lm-evaluation-harness
# 0.4.13 installs for the Minerva MATH scorer:
#   math_verify 0.9.0 -> latex2sympy2_extended 1.11.0 -> antlr4-python3-runtime 4.11.0
# The antlr pin matters: sympy's LaTeX parser and the generated parser bundled in
# latex2sympy2_extended are both built for the 4.11 runtime.
final: prev:
let
  pythonOverrides = pyFinal: _pyPrev: {
    antlr4-python3-runtime = pyFinal.buildPythonPackage {
      pname = "antlr4-python3-runtime";
      version = "4.11.0";
      format = "wheel";
      src = final.fetchurl {
        url = "https://files.pythonhosted.org/packages/37/f3/2cab1ffe441ffcc4605bf638f524fed86db6699a25d04bbecb5bfdde372b/antlr4_python3_runtime-4.11.0-py3-none-any.whl";
        hash = "sha256-9SP5E4coMEXxKcNDs2Oo3aOgfupichuO2pXy2LgXtlY=";
      };
      pythonImportsCheck = [ "antlr4" ];
    };

    latex2sympy2-extended = pyFinal.buildPythonPackage {
      pname = "latex2sympy2-extended";
      version = "1.11.0";
      format = "wheel";
      src = final.fetchurl {
        url = "https://files.pythonhosted.org/packages/e9/61/f75cd1fa54d8434276126034aed54dd120747de9a8fa013cdd79545ccbeb/latex2sympy2_extended-1.11.0-py3-none-any.whl";
        hash = "sha256-rrt31SziaeJQKOS+qJ3bFNJCuja897Y2SW+1/Zco0jQ=";
      };
      dependencies = [
        pyFinal.sympy
        pyFinal.antlr4-python3-runtime
      ];
      pythonImportsCheck = [ "latex2sympy2_extended" ];
    };

    math-verify = pyFinal.buildPythonPackage {
      pname = "math-verify";
      version = "0.9.0";
      format = "wheel";
      src = final.fetchurl {
        url = "https://files.pythonhosted.org/packages/62/76/6b4969bccc842b6567f7e6ee015684b9428a9b7fcbdf479e73716f43597f/math_verify-0.9.0-py3-none-any.whl";
        hash = "sha256-NwPnxIhTVAJ/qEQJ12KllqKQbR/U3reDYYdr2QWnYZQ=";
      };
      dependencies = [ pyFinal.latex2sympy2-extended ];
      pythonImportsCheck = [ "math_verify" ];
    };
  };

  python3 = prev.python3.override { packageOverrides = pythonOverrides; };
in
{
  inherit python3;
  python3Packages = python3.pkgs;
}
