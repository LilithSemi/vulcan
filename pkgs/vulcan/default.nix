{
  lib,
  flakever,
  stdenv,
  mkShell,
  wrapCCWith,
  zig,
  lld,
  binutils,
  etsoc-sysemu,
}:
let
  inherit (stdenv) targetPlatform hostPlatform;

  targetPrefix = lib.optionalString (targetPlatform != hostPlatform) (targetPlatform.config + "-");
in
stdenv.mkDerivation (finalAttrs: {
  pname = "vulcan";
  inherit (flakever) version;

  src = lib.cleanSource ../../.;

  nativeBuildInputs = [ zig ];

  zigDeps = zig.fetchDeps {
    inherit (finalAttrs) pname version src;
    hash = "sha256-zCP3Yk9/byB0kvyTQT24m27o22CJ78YH5n+c8J1WD/Y=";
  };

  postConfigure = ''
    ln -s $zigDeps $ZIG_GLOBAL_CACHE_DIR/p
  '';

  doCheck = true;
  nativeCheckInputs = lib.optional etsoc-sysemu.meta.available etsoc-sysemu;

  # Needed so `wrapCCWith` works, remove when nixpkgs supports `vcc`
  postInstall = ''
    ln -s $out/bin/vcc $out/bin/${targetPrefix}gcc
  '';

  passthru = {
    shell = mkShell {
      name = "vulcan-dev-shell";

      packages = [
        zig
        lld
        binutils
      ];
    };

    cc = wrapCCWith {
      cc = finalAttrs.finalPackage;
    };

    vcc-stdenv = stdenv.override {
      inherit (finalAttrs.finalPackage) cc;
      allowedRequisites = stdenv.allowedRequisites ++ [
        finalAttrs.finalPackage
        finalAttrs.finalPackage.cc.expand-response-params
        finalAttrs.finalPackage.cc.bintools
      ];
    };
  };
})
