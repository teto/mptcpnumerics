{ pkgs ?  import <nixpkgs> {} }:

let
  prog = pkgs.python3Packages.callPackage ./mptcpnumerics.nix {};
in
  prog
