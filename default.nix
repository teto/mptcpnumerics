{ pkgs ? import <nixpkgs> {} }:
# with (import <nixpkgs>);

with pkgs;
let
  pl = python3Packages.buildPythonApplication rec {

  pname="mptcpnumerics";
  version="dev";
  src = ./.;
  propagatedBuildInputs= with python3Packages; [
    sympy
    matplotlib
    sortedcontainers
    pulp
    ];


  doCheck=false;

  # meta = with stdenv.lib; {
  #   homepage = http://lostpackets.de/khal/;
  #   description = "test proogram";
  #   license = licenses.gpl;
  # };
};
in pl
