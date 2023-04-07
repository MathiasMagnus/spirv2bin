# spirv2bin **(pre-alpha)**

spirv2bin is a pre-alpha OpenCL SPIR-V to vendor binary translator. It's main purpose is two-fold: retrofit SPIR-V unaware runtimes with SPIR-V ingestion capabilities and to provide the bulk of application logic required for said runtimes to learn consuming SPIR-V.

The project initially targets AMD's OpenCL runtime.

## Dependencies

For the (soon to be) library part:

- llvm-project
- ROCm-Device-Libs
- ROCm-CompilerSupport
- SPIRV-Headers
- SPIRV-LLVM-Translator

For the (soon to be) OpenCL layer part:

- OpenCL-Headers

For the examples:

- OpenCL-ICD-Loader

### Notes on deps

- In it's current pre-proof-of-concept (POC) stage the library has not been factored out and the layer does not exist. Everything's linked to an example executable.
- The ROCm deps wouldn't strictly be necessary, but having tried writing a minimal compiler and linker driver inside the library, using AMD's code object manager (comgr) instead is a _huge_ convenience.
  - AMD outsourcing these features to a library which is then used by all their offload APIs is an example to follow.

### Convenience project for deps

Because much of the work up until this stage was trying to find compatible combinations of dependencies, and experimenting with them in a plug'n'play nature, a convenience project was created that defines just the dep git URLs, hashes/tags and builds all the dependencies with out-of-tree patches in the correct order.

To avoid setup twice, once for local development once for CI runs, it's using the cmake-presets CLI interface to build all deps. ExternalProject can't relay preset invocations to external projects, so the process is entirely overriden by custom commands on every step.

## Disclaimer

I am not a compiler developer (yet, anyway), my knowledge of compiler front-ends and back-ends are mostly theoretical. I'm hoping that others with relevant knowledge can spare a few cycles, at least to provide pointers on how solutions look like or where to start looking.

## Current blocker

For a detail on the current blocker, refer to [poc executable README](examples/comgr-test/README.md).

## Future plans

1. Reach a proof-of-concept (POC) stage, translating SPIR-V to AMDGPU and feeding it to a production OpenCL runtime to execute.
2. Split the POC into a library and executable.
3. Wrap the library into a proper OpenCL-Layer for easy use.

Future directions are up to upstream (AMD) and community interest.

- Adding the translation logic to [ROCm-OpenCL-Runtime](https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime) is a less involved way to upstream SPIR-V ingestion to AMD's OpenCL Runtime. The library logic can be taken almost as-is.
- Adding the translation logic to [ROCm-CompilerSupport](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport) (aka. Code Object Manager, or comgr) multiple runtimes could be taught how to consume SPIR-V for Compute.
  - This would reduce the code required in the OpenCL runtime to expose SPIR-V in a conforming manner to ~100 lines (extra entry point, few extra valid queries, return/consume IL binary).
  - This could remedy the humongous shortcoming of HIP, namely that AMD lacks its own virtual ISA. If comgr learns to consume SPIR-V and translate it to gfxXYZ (arch-specific AMDGPU), SPIR-V could be embedded into executables (much like PTX is for CUDA) and have the runtimme JIT compile kernels on startup, before submission, or whatever. The value proposition is there, but I suspect it would still require some plumbing from the compiler side.
    - Some compile-time constants such as wave/warp size, which are arch dependent (come from `--offload-arch=`) need extra care when a virtual ISA is targeted. These constants would need to exposed in an alternative way.
- NVIDIA support depends on reliably being able to determine the `sm_XY` ISA version of a device.
  - Potentially adding PTX output through comgr would enable some interesting use-cases.
