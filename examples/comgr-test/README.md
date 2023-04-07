# Proof of concept

This is the proof-of-concept executable, which intends on doing the following:

1. Select a platform/device based on CLI input, or choose the default ones.
2. Compile a dummy kernel (SAXPY) on the selected device, only to extract the runtime-specific binary.
3. Give the binary to comgr to query the ISA name used by the device.
4. Load the actual SPIR-V we want to run (also SAXPY, but could be anything) from disk and run it through: the SPIRV-LLVM-Translator and call the AMDGPU back-end of LLVM using the previously queries ISA name.
5. Feed the reuslting LLVM BC to comgr and link it to the apporpriate device libraries.
6. Compile to a functional ELF binary for uploading to the device.
7. Feed the AMDGPU binary to the real OpenCL runtime as a binary for execution.

## Current issue

For some unknown reason, the BC fed to comgr can't be linked to the device libraries. When patching LLVM using [002-add-cerr-tracing-of-linking-error.patch](../../external/patches/llvm-project/002-add-cerr-tracing-of-linking-error.patch) with the specific version source and version of deps as found in [the deps project](../../external/CMakeLists.txt) the application produces the following command-line output:

```none
C:\Users\mate\Source\Repos\spirv2bin\.vscode\build\msbuild-msvc-v143\exmples\comgr-test\bin\Release\OpenCL-Cpp-SAXPY.exe
Selected platform: AMD Accelerated Parallel Processing
Selected device: gfx1032
Compiling dummy kernel: "C:\\Users\\mate\\Source\\Repos\\spirv2bin\\.vscode\\build\\msbuild-msvc-v143\\exmples\\comgr-test\\bin\\Release\\OpenCL-Cpp-SAXPY.cl"
ISA name obtained from comgr: amdgcn-amd-amdhsa--gfx1032
Triple as parsed from ISA name: amdgcn-amd--amdhsa---
gfx as parsed from ISA name: gfx1032
CanonCPUName: gfx1032
device_libs_dataset has 11 AMD_COMGR_DATA_KIND_BC elements.
Processing dataset element.
I was here.
FunctionLink error 1.
Materialize error 2.
IRLink error 2.
Link error 7.
Error 3.
amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC)
```

The logic of how an end-to-end compilation workflow looks like using comgr was inspired by [comgr's compile_minimal_test.c](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/blob/amd-stg-open/lib/comgr/test/compile_minimal_test.c), that logic is shortened, because the entry point of compilation here starts with the bitcode produced by previous steps.

While the comments inside LLVM are of commendable detail, they don't speak much to an outsider. With my handwavy understanding of [what does "materialize" mean in llvm GlobalValue.h](https://stackoverflow.com/questions/45642228/what-does-materialize-mean-in-llvm-globalvalue-h), I can't really figure out which bitcode element fails materialization in the supplied `OpenCL-Cpp-SAXPY.spv`. It was produced using Clang 16 via: `clang.exe --target=spirv64 -fintegrated-objemitter .\OpenCL-Cpp-SAXPY.cl -o .\OpenCL-Cpp-SAXPY.spv`

## comgr.hpp

Because it's a lot of lines of code for my taste, I started creating a C++ wrapper for comgr. The general API design resembles OpenCL a lot, so similar RAII lifetime handling and error reporting are easy to implement.
