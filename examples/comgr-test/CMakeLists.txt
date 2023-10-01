cmake_minimum_required(VERSION 3.7)
cmake_policy(VERSION 3.7...3.25)

project(OpenCL-Cpp-SAXPY
  LANGUAGES CXX
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" CACHE PATH "asdf")

if(NOT TARGET OpenCL::OpenCL)
  find_package(OpenCL REQUIRED CONFIG
    COMPONENTS
      OpenCL
      HeadersCpp
      UtilsCpp
  )
endif()

find_package(Threads REQUIRED)
find_package(amd_comgr REQUIRED)
find_package(LLVM 16.0.0 REQUIRED)
find_package(SPIRV-Headers REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LLVM_SPIRV REQUIRED LLVMSPIRVLib)

set(Sources ${PROJECT_NAME}.cpp)
set(Kernels
  ${PROJECT_NAME}.cl
  ${PROJECT_NAME}.spv
)

add_executable(${PROJECT_NAME}
  ${Sources}
  ${Kernels}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE
    cxx_std_20
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${LLVM_SPIRV_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME}
  PRIVATE
    ${LLVM_SPIRV_LIBRARY_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    OpenCL::HeadersCpp
    OpenCL::UtilsCpp
    Threads::Threads
    amd_comgr
    ${LLVM_SPIRV_LINK_LIBRARIES}
    SPIRV-Headers::SPIRV-Headers
    clangAnalysis
    clangARCMigrate
    clangAST
    clangASTMatchers
    clangBasic
    clangCodeGen
    clangCrossTU
    clangDriver
    clangEdit
    clangExtractAPI
    clangFormat
    clangFrontend
    clangFrontendTool
    clangIndex
    clangLex
    clangParse
    clangRewrite
    clangRewriteFrontend
    clangSema
    clangSerialization
    clangStaticAnalyzerCheckers
    clangStaticAnalyzerCore
    clangStaticAnalyzerFrontend
    clangSupport
    clangToolingCore
    clangToolingInclusions
    lldCommon
    lldELF
    LLVMWindowsManifest
    LLVMXRay
    LLVMLibDriver
    LLVMDlltoolDriver
    LLVMCoverage
    LLVMLineEditor
    LLVMXCoreDisassembler
    LLVMXCoreCodeGen
    LLVMXCoreDesc
    LLVMXCoreInfo
    LLVMX86TargetMCA
    LLVMX86Disassembler
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMWebAssemblyDisassembler
    LLVMWebAssemblyAsmParser
    LLVMWebAssemblyCodeGen
    LLVMWebAssemblyDesc
    LLVMWebAssemblyUtils
    LLVMWebAssemblyInfo
    LLVMVEDisassembler
    LLVMVEAsmParser
    LLVMVECodeGen
    LLVMVEDesc
    LLVMVEInfo
    LLVMSystemZDisassembler
    LLVMSystemZAsmParser
    LLVMSystemZCodeGen
    LLVMSystemZDesc
    LLVMSystemZInfo
    LLVMSparcDisassembler
    LLVMSparcAsmParser
    LLVMSparcCodeGen
    LLVMSparcDesc
    LLVMSparcInfo
    LLVMRISCVTargetMCA
    LLVMRISCVDisassembler
    LLVMRISCVAsmParser
    LLVMRISCVCodeGen
    LLVMRISCVDesc
    LLVMRISCVInfo
    LLVMPowerPCDisassembler
    LLVMPowerPCAsmParser
    LLVMPowerPCCodeGen
    LLVMPowerPCDesc
    LLVMPowerPCInfo
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMMSP430Disassembler
    LLVMMSP430AsmParser
    LLVMMSP430CodeGen
    LLVMMSP430Desc
    LLVMMSP430Info
    LLVMMipsDisassembler
    LLVMMipsAsmParser
    LLVMMipsCodeGen
    LLVMMipsDesc
    LLVMMipsInfo
    LLVMLoongArchDisassembler
    LLVMLoongArchAsmParser
    LLVMLoongArchCodeGen
    LLVMLoongArchDesc
    LLVMLoongArchInfo
    LLVMLanaiDisassembler
    LLVMLanaiCodeGen
    LLVMLanaiAsmParser
    LLVMLanaiDesc
    LLVMLanaiInfo
    LLVMHexagonDisassembler
    LLVMHexagonCodeGen
    LLVMHexagonAsmParser
    LLVMHexagonDesc
    LLVMHexagonInfo
    LLVMBPFDisassembler
    LLVMBPFAsmParser
    LLVMBPFCodeGen
    LLVMBPFDesc
    LLVMBPFInfo
    LLVMAVRDisassembler
    LLVMAVRAsmParser
    LLVMAVRCodeGen
    LLVMAVRDesc
    LLVMAVRInfo
    LLVMARMDisassembler
    LLVMARMAsmParser
    LLVMARMCodeGen
    LLVMARMDesc
    LLVMARMUtils
    LLVMARMInfo
    LLVMAMDGPUTargetMCA
    LLVMAMDGPUDisassembler
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUUtils
    LLVMAMDGPUInfo
    LLVMAArch64Disassembler
    LLVMAArch64AsmParser
    LLVMAArch64CodeGen
    LLVMAArch64Desc
    LLVMAArch64Utils
    LLVMAArch64Info
    LLVMOrcJIT
    LLVMWindowsDriver
    LLVMMCJIT
    LLVMJITLink
    LLVMInterpreter
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMOrcTargetProcess
    LLVMOrcShared
    LLVMDWP
    LLVMDebugInfoLogicalView
    LLVMDebugInfoGSYM
    LLVMOption
    LLVMObjectYAML
    LLVMObjCopy
    LLVMMCA
    LLVMMCDisassembler
    LLVMLTO
    LLVMCFGuard
    LLVMFrontendOpenACC
    LLVMFrontendHLSL
    LLVMExtensions
    Polly
    PollyISL
    LLVMPasses
    LLVMCoroutines
    LLVMipo
    LLVMInstrumentation
    LLVMVectorize
    LLVMLinker
    LLVMFrontendOpenMP
    LLVMDWARFLinkerParallel
    LLVMDWARFLinker
    LLVMGlobalISel
    LLVMMIRParser
    LLVMAsmPrinter
    LLVMSelectionDAG
    LLVMCodeGen
    LLVMObjCARCOpts
    LLVMIRPrinter
    LLVMInterfaceStub
    LLVMFileCheck
    LLVMFuzzMutate
    LLVMTarget
    LLVMScalarOpts
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMTransformUtils
    LLVMBitWriter
    LLVMAnalysis
    LLVMProfileData
    LLVMSymbolize
    LLVMDebugInfoPDB
    LLVMDebugInfoMSF
    LLVMDebugInfoDWARF
    LLVMObject
    LLVMTextAPI
    LLVMMCParser
    LLVMIRReader
    LLVMAsmParser
    LLVMMC
    LLVMDebugInfoCodeView
    LLVMBitReader
    LLVMFuzzerCLI
    LLVMCore
    LLVMRemarks
    LLVMBitstreamReader
    LLVMBinaryFormat
    LLVMTargetParser
    LLVMTableGen
    LLVMSupport
    LLVMDemangle
)

target_link_directories(${PROJECT_NAME}
  PRIVATE
    "${LLVM_LIBRARY_DIRS}"
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS
    AMD_COMGR_IMPORT_DECORATOR=
)

source_group("Sources" FILES ${Sources})
source_group("Kernels" FILES ${Kernels})

# The following CMake snippet copies kernel files next to the executable in the build tree.
# It does the following steps:
#
# 1. Checks whether the generator at hand is a multi-config generator or not.
#
# This is needed because add_custom_command & add_custom_target don't accept target-dependent
# generator expressions, so we can't use $<TARGET_FILE_DIR:tgt> to ask our tagrget which
# folder the executable goes to. Therefore we handle the folder difference between single-
# and multi-config generators ourselves. We assemble the path using CMAKE_RUNTIME_OUTPUT_DIRECTORY
# which is the variable used as the default for TARGET_FILE_DIR. If we alter TARGET_FILE_DIR to
# differ from the default, this snippet won't work without modification.
#
# 2. Sets up helper vars for a) what TO_COPY b) where we COPY_TO c) OUTPUTs the helper target
# DEPENDS on.
#
# CMake stores/formats lists as a semi-colon delimited string. The helper vars transform the
# ${Kernels} variable which for eg. may be k1.cl;k2.cl . This works because add_executable awaits
# source files either via absolute path or relative paths to CMAKE_CURRENT_SOURCE_DIR. We want to
# transform k1.cl;k2.cl to <dir>/k1.cl;<dir>/k2.cl where <dir> may be CMAKE_RUNTIME_OUTPUT_DIRECTORY or
# CMAKE_CURRENT_SOURCE_DIR depending on whether we're assembling paths to sources or destinations.
# $<JOIN:list,delimiter> will join the elements of 'list' using the string 'delimiter'. The delimiter
# isn't prepended to the first item, we have to specify that outselves, hence the complete expression
# takes the form <dir>/$<JOIN:list,;<dir>/>.
#
# 3. Registers a custom command which copies all files in ${Kernels} to the proper build tree folder
#
# 4. Registers a custom target which DEPENDS on the OUTPUT of the custom command
#
# 5. Sets up the executable to depend on kernel code deployment, so building our target also copies
# potentially updated kernel code, therefore if someone isn't building the 'all' meta-target, but
# the executable only, they won't miss kernel-code updates.
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/$<JOIN:${Kernels},;${CMAKE_CURRENT_SOURCE_DIR}/>")
set(COPY_TO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")
set(OUTPUT_AND_DEPENDS "${COPY_TO}/$<JOIN:${Kernels},;${COPY_TO}/>")
add_custom_command(
  OUTPUT "${OUTPUT_AND_DEPENDS}"
  COMMAND ${CMAKE_COMMAND}
  ARGS
  -E copy_if_different
  "${TO_COPY}"
  "${COPY_TO}"
  COMMENT "Copying CL kernels for ${PROJECT_NAME}"
  COMMAND_EXPAND_LISTS
  DEPENDS ${Kernels}
)
add_custom_target(${PROJECT_NAME}-device-code
  DEPENDS "${OUTPUT_AND_DEPENDS}"
)
add_dependencies(${PROJECT_NAME}
  ${PROJECT_NAME}-device-code
)
