find_package(Git REQUIRED QUIET)

if(NOT DEFINED DIRECTORY OR NOT DEFINED PATCHES)
    message(FATAL_ERROR "DIRECTORY and PATCHES are mandatory.")
endif()

foreach(PATCH IN LISTS PATCHES)
    # First, check if the patch is already applied
    execute_process(
        COMMAND
        "${GIT_EXECUTABLE}"
        "apply"
        "--check"
        "--reverse"
        "--directory=${ARG_DIRECTORY}"
        "--unsafe-paths"
        "--ignore-whitespace"
        "${PATCH}"
        WORKING_DIRECTORY "${ARG_DIRECTORY}"
        RESULT_VARIABLE GIT_RESULT
        OUTPUT_VARIABLE GIT_OUTPUT
        ERROR_VARIABLE GIT_ERROR
    )

    if(GIT_RESULT STREQUAL "0")
        message(VERBOSE "${PATCH} need not be applied.")
        continue()
    endif()

    execute_process(
        COMMAND
        "${GIT_EXECUTABLE}"
        "apply"
        "--directory=${ARG_DIRECTORY}"
        "--unsafe-paths"
        "--ignore-whitespace"
        "${PATCH}"
        WORKING_DIRECTORY "${ARG_DIRECTORY}"
        RESULT_VARIABLE GIT_RESULT
        OUTPUT_VARIABLE GIT_OUTPUT
        ERROR_VARIABLE GIT_ERROR
    )

    if(NOT GIT_RESULT STREQUAL "0")
        message(FATAL_ERROR "${GIT_ERROR}")
    else()
        message(VERBOSE "${PATCH} applied.")
    endif()
endforeach()
