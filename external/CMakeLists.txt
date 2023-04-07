cmake_minimum_required(VERSION 3.25)
cmake_policy(VERSION 3.25)

project(spirv2bin_dependencies LANGUAGES NONE)

find_package(Git REQUIRED)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_BASE "${PROJECT_BINARY_DIR}")

function(preset_external_project)
    cmake_parse_arguments(ARG
        "" # options
        "PROJECT;URL;TAG;SOURCE_SUBDIR" # oneValueArgs
        "PATCHES;DEPENDS" # multiValueArgs
        ${ARGN}
    )
    set(EP_${ARG_PROJECT}_COPY_PRESET
        "${CMAKE_COMMAND}"
        "-E"
        "copy_if_different"
        "${PROJECT_SOURCE_DIR}/CMakeUserPresets.json"
        "${PROJECT_BINARY_DIR}/Source/${ARG_PROJECT}/${ARG_SOURCE_SUBDIR}/CMakeUserPresets.json"
    )
    list(JOIN ARG_PATCHES "$<SEMICOLON>" JOINED_PATCHES)
    set(EP_${ARG_PROJECT}_PATCH_COMMAND
        "${CMAKE_COMMAND}"
        "--log-level=VERBOSE"
        "-D"
        "DIRECTORY=${PROJECT_BINARY_DIR}/Source/${ARG_PROJECT}"
        "-D"
        "PATCHES=${JOINED_PATCHES}"
        "-P"
        "${PROJECT_SOURCE_DIR}/GitApplyOrSucceedIfAlreadyApplied.cmake"
    )
    set(EP_${ARG_PROJECT}_CONFIGURE_COMMAND
        "${CMAKE_COMMAND}"
        "--preset"
        "${ARG_PROJECT}"
    )
    set(EP_${ARG_PROJECT}_BUILD_COMMAND
        "${CMAKE_COMMAND}"
        "--build"
        "--preset"
        "${ARG_PROJECT}"
    )
    set(EP_${ARG_PROJECT}_INSTALL_COMMAND
        "${CMAKE_COMMAND}"
        "--build"
        "--preset"
        "${ARG_PROJECT}-install"
    )

    if(DEFINED ARG_DEPENDS)
        set(DEPENDS_ARG DEPENDS ${ARG_DEPENDS})
    else()
        set(DEPENDS_ARG)
    endif()

    ExternalProject_Add(${ARG_PROJECT}
        GIT_REPOSITORY ${ARG_URL}
        GIT_TAG ${ARG_TAG}
        GIT_SHALLOW ON
        UPDATE_DISCONNECTED OFF
        CONFIGURE_HANDLED_BY_BUILD ON
        BINARY_DIR "${PROJECT_BINARY_DIR}/Source/${ARG_PROJECT}/${ARG_SOURCE_SUBDIR}"
        PATCH_COMMAND "${EP_${ARG_PROJECT}_PATCH_COMMAND}"
        CONFIGURE_COMMAND "${EP_${ARG_PROJECT}_CONFIGURE_COMMAND}"
        BUILD_COMMAND "${EP_${ARG_PROJECT}_BUILD_COMMAND}"
        INSTALL_COMMAND "${EP_${ARG_PROJECT}_INSTALL_COMMAND}"
        ${DEPENDS_ARG}
    )
    add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/Source/${ARG_PROJECT}/${ARG_SOURCE_SUBDIR}/CMakeUserPresets.json"
        DEPENDS "${PROJECT_SOURCE_DIR}/CMakeUserPresets.json"
        COMMAND ${EP_${ARG_PROJECT}_COPY_PRESET}
    )
    add_custom_target(${ARG_PROJECT}-copy-preset
        DEPENDS "${PROJECT_BINARY_DIR}/Source/${ARG_PROJECT}/${ARG_SOURCE_SUBDIR}/CMakeUserPresets.json"
    )
    add_dependencies(${ARG_PROJECT} ${ARG_PROJECT}-copy-preset)
endfunction(preset_external_project)

preset_external_project(
    PROJECT llvm-project
    URL https://github.com/llvm/llvm-project.git
    TAG 08d094a0e457360ad8b94b017d2dc277e697ca76 # llvmorg-16.0.0
    SOURCE_SUBDIR llvm
    PATCHES "${PROJECT_SOURCE_DIR}/patches/llvm-project/001-narrowing-conversion.patch"
)
preset_external_project(
    PROJECT ROCm-Device-Libs
    URL https://github.com/Mystro256/ROCm-Device-Libs.git
    TAG b000631a2d331133e07d108ca0c81d80d7d8fb39 # release/16.x @ 2023.03.30.
    SOURCE_SUBDIR "."
    PATCHES "${PROJECT_SOURCE_DIR}/patches/ROCm-Device-Libs/001-depend-on-pre-built-llvm.patch"
    DEPENDS llvm-project
)
preset_external_project(
    PROJECT ROCm-CompilerSupport
    URL https://github.com/Mystro256/ROCm-CompilerSupport.git
    TAG e3b172aebf43c84364caec372833bef9897517e4 # release/16.x @ 2023.03.30.
    SOURCE_SUBDIR "lib/comgr"
    PATCHES
        "${PROJECT_SOURCE_DIR}/patches/ROCm-CompilerSupport/001-depend-on-pre-built-llvm.patch"
        "${PROJECT_SOURCE_DIR}/patches/ROCm-CompilerSupport/002-fix-static-lib-builds.patch"
        "${PROJECT_SOURCE_DIR}/patches/ROCm-CompilerSupport/003-msvc-reduce-noise-on-console.patch"
    DEPENDS
        llvm-project
        ROCm-Device-Libs
)
preset_external_project(
    PROJECT SPIRV-Headers
    URL https://github.com/KhronosGroup/SPIRV-Headers.git
    TAG 1feaf4414eb2b353764d01d88f8aa4bcc67b60db # sdk-1.3.243.0 @ 2023.04.02.
    SOURCE_SUBDIR "."
)
preset_external_project(
    PROJECT SPIRV-LLVM-Translator
    URL https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    TAG 794e180083c9ba1b61a1a63ba40bb729dbad0c0e # llvm_release_160 @ 2023.04.02.
    SOURCE_SUBDIR "."
    DEPENDS
        SPIRV-Headers
        llvm-project
)
