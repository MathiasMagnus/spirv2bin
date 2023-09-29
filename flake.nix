{
  description = "SPIR-V shenanigans";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = { allowBroken = true; };
    };
  in rec {
    packages.${system} = {
      opencl-sdk = pkgs.callPackage ({
        stdenv,
        fetchFromGitHub,
        cmake,
        ninja,
        git
      }: stdenv.mkDerivation {
        name = "opencl-sdk";
        version = "1.0";

        nativeBuildInputs = [
          cmake
          ninja
          git
        ];

        cmakeFlags = [
          "-DOPENCL_SDK_BUILD_SAMPLES=OFF"
        ];

        src = fetchFromGitHub {
          owner = "KhronosGroup";
          repo = "OpenCL-SDK";
          rev = "cd612e53cc3ffdf28a13634c28212fb6ee57ba8b";
          hash = "sha256-8XRR3QR5pU6kRwYj5fI2sfJkfN3lux+BPWW+lbqFVbc=";
          fetchSubmodules = true;
        };
      }) {};

      rocm-device-libs = pkgs.callPackage ({
        stdenv,
        cmake,
        llvmPackages_16,
        fetchFromGitHub
      }: stdenv.mkDerivation rec {
        name = "rocm-device-libs";
        version = "5.7.0";

        nativeBuildInputs = [
          cmake
        ];

        patches = pkgs.rocm-device-libs.patches;

        buildInputs = [
          llvmPackages_16.llvm
          llvmPackages_16.clang
        ];

        src = fetchFromGitHub {
          owner = "RadeonOpenCompute";
          repo = "ROCm-Device-Libs";
          rev = "rocm-${version}";
          hash = "sha256-f6/LAhJ2mBDO1/JloHvl7MJyDo3WutbXd4IDknA9nzM=";
        };

        postPatch = ''
          substituteInPlace cmake/OCL.cmake test/compile/CMakeLists.txt \
            --replace "\''$<TARGET_FILE:clang>" "${llvmPackages_16.clang}/bin/clang" \
            --replace "-Werror" ""

          substituteInPlace ockl/src/dots.cl \
            --replace dot10-insts dot7-insts
        '';

        doTest = false;
      }) {};

      rocm-comgr = pkgs.callPackage ({
        stdenv,
        llvmPackages_16,
        cmake,
        fetchFromGitHub
      }: stdenv.mkDerivation rec {
        name = "rocm-comgr";
        version = "5.7.0";

        nativeBuildInputs = [
          cmake
        ];

        buildInputs = [
          packages.${system}.rocm-device-libs
          llvmPackages_16.clang-unwrapped.dev
          llvmPackages_16.llvm
          llvmPackages_16.lld
        ];

        src = fetchFromGitHub {
          owner = "RadeonOpenCompute";
          repo = "ROCm-CompilerSupport";
          rev = "rocm-${version}";
          hash = "sha256-QB3G0V92UTW67hD6+zSuExN1+eMT820iYSlMyZeWSFw=";
        };

        sourceRoot = "${src.name}/lib/comgr";

        postPatch = ''
          substituteInPlace cmake/opencl_pch.cmake \
            --replace "\''${CLANG_CMAKE_DIR}/../../../*/opencl-c.h" "${llvmPackages_16.clang-unwrapped.lib}/lib/clang/16/include/opencl-c.h" \
            --replace "< \''${OPENCL_C_H}" "-I${llvmPackages_16.clang-unwrapped.lib}/lib/clang/16/include < \''${OPENCL_C_H}"

          substituteInPlace src/comgr-isa-metadata.def \
            --replace EF_AMDGPU_MACH_AMDGCN_GFX941 EF_AMDGPU_MACH_AMDGCN_GFX940 \
            --replace EF_AMDGPU_MACH_AMDGCN_GFX942 EF_AMDGPU_MACH_AMDGCN_GFX940
        '';

      }) {};
      spirv-llvm-translator = (pkgs.spirv-llvm-translator.override {
        inherit (pkgs.llvmPackages_16) llvm;
      }).overrideAttrs (old: {
        postPatch = ''
          substituteInPlace lib/SPIRV/*.cpp \
            --replace SPIR_FUNC Fast
        '';
      });
    };

    devShells.${system}.default = let
      llvm = pkgs.llvmPackages_16;
    in pkgs.clang16Stdenv.mkDerivation {
      name = "spirv";
      nativeBuildInputs = [
        packages.${system}.opencl-sdk
        packages.${system}.rocm-comgr
        packages.${system}.spirv-llvm-translator
        llvm.clang
        llvm.llvm
        llvm.lld
        pkgs.spirv-tools
        pkgs.cmake
        pkgs.ninja
        pkgs.clinfo
        pkgs.spirv-headers
        pkgs.pkg-config
        pkgs.rocm-opencl-runtime
        pkgs.rocm-opencl-icd
        pkgs.libxml2
        pkgs.libffi
      ];

      OCL_ICD_VENDORS = "${pkgs.rocm-opencl-icd}/etc/OpenCL/vendors";
    };
  };
}
