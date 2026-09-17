# Build of the Italian draughts C++ engine.
#
# Without LibTorch (no external dependencies, everything compiles and the tests run):
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# With LibTorch (needed to use a real neural network / the GPU):
#   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DWITH_LIBTORCH=ON \
#         -DCMAKE_PREFIX_PATH=/path/to/libtorch
#   cmake --build build -j
#
# If the GPU is not detected, or the build runs on a machine without one, the
# architecture can be declared by hand:
#         -DTORCH_CUDA_ARCH_LIST=8.9
#
# ON WINDOWS, two differences that cannot be guessed:
#   1. the default generator (Visual Studio) is MULTI-CONFIGURATION, so
#      CMAKE_BUILD_TYPE is IGNORED and the configuration must be given at build
#      time, not at configure time:
#         cmake --build build --config Release
#      Leaving it out builds in Debug: an engine several times slower, with
#      nothing to flag it;
#   2. the executable ends up in build\Release\, not in build\.
#   LibTorch for Windows is needed (the "release" variant, consistent with MSVC)
#   and a Visual Studio with the C++ tools; MinGW does not work, because the
#   LibTorch libraries are built with MSVC.
#
# The g++ scripts (build.sh, and build.ps1 without LibTorch) need no CMake and
# build and run every test: they remain the simplest path for everything except
# the LibTorch backend.
cmake_minimum_required(VERSION 3.20)
project(dama_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(WITH_LIBTORCH "Build the LibTorch backend (needed for a neural network/GPU)" OFF)

find_package(Threads REQUIRED)

add_library(dama_core INTERFACE)
target_include_directories(dama_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dama_core INTERFACE Threads::Threads)

if(WITH_LIBTORCH)
    # CUDA architectures: declared, not left to LibTorch to detect.
    #
    # Left alone, the LibTorch configuration tries to DETECT the GPU by
    # compiling a small CUDA program on the fly. That test project inherits
    # CMAKE_CXX_STANDARD=20 from here, so it asks for the "CUDA20" dialect, and
    # any CMake older than CUDA 12 does not know which flags enable it: the
    # configuration stops with
    #     Target "cmTC_xxxxx" requires the language dialect "CUDA20"
    # It really happened, on Ubuntu 22.04 (CMake 3.22) with CUDA 12.4.
    #
    # Detection is useless here anyway: this project contains not a single CUDA
    # file and uses only LibTorch's prebuilt libraries. Saying which
    # architecture matters is enough for the step to be skipped. nvidia-smi is
    # asked, with a fallback list if it does not answer (for example when
    # building on a machine without a GPU).
    if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _nvsmi_out OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _nvsmi_rc)
        set(_cc "")
        if(_nvsmi_rc EQUAL 0)
            string(REGEX MATCH "[0-9]+\\.[0-9]+" _cc "${_nvsmi_out}")
        endif()
        if(_cc)
            set(TORCH_CUDA_ARCH_LIST "${_cc}")
            message(STATUS "CUDA architecture detected by nvidia-smi: ${_cc}")
        else()
            set(TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6;8.9;9.0")
            message(STATUS "nvidia-smi not available: fallback CUDA "
                           "architectures ${TORCH_CUDA_ARCH_LIST}")
        endif()
    endif()
    find_package(Torch REQUIRED)
    # TORCH_CXX_FLAGS carries -D_GLIBCXX_USE_CXX11_ABI=<0|1>, the ABI LibTorch
    # was built with. Leaving it out is the classic mistake: the build seems to
    # work and then linking fails with "undefined symbol" on
    # std::__cxx11::basic_string. It MUST go into the global flags.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_compile_definitions(dama_core INTERFACE WITH_LIBTORCH)
    target_link_libraries(dama_core INTERFACE "${TORCH_LIBRARIES}")
    message(STATUS "LibTorch enabled: ${TORCH_LIBRARIES}")
    message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "LibTorch NOT enabled: the engine will use the fake backend only")
endif()

# --- main executable ---
add_executable(dama_engine main.cpp)
target_link_libraries(dama_engine PRIVATE dama_core)

# --- validation tools ---
foreach(tool perft encoder_parity mcts_selfplay_test threading_test selfplay_mt_test arena_test)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} PRIVATE dama_core)
endforeach()

if(WITH_LIBTORCH)
    add_executable(parity_check parity_check.cpp)
    target_link_libraries(parity_check PRIVATE dama_core)

    # WINDOWS: the LibTorch DLLs must be copied NEXT TO the executable.
    #
    # On Linux the library path is written into the binary (rpath) and the
    # program starts. On Windows there is no equivalent: the loader looks for
    # DLLs in the executable's folder and on the PATH, and that is all. Without
    # this copy the .exe compiles and links without a warning, then at startup
    # it dies BEFORE main with code 0xc0000135 and no message: no error line, no
    # trace, just a process that does not start. It is the most unpleasant way
    # to discover a build problem, and also the most common one with LibTorch on
    # Windows.
    #
    # copy_if_different, not copy: these are a few gigabytes of DLLs, and
    # copying them again at every build would make every single compilation
    # unbearable.
    if(WIN32)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        foreach(exe dama_engine parity_check)
            add_custom_command(TARGET ${exe} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS} $<TARGET_FILE_DIR:${exe}>
                COMMENT "copying the LibTorch DLLs next to ${exe}")
        endforeach()
    endif()
endif()

# --- tests (ctest) ---
enable_testing()
add_test(NAME perft COMMAND perft)
add_test(NAME encoder_parity COMMAND encoder_parity)
add_test(NAME mcts_selfplay COMMAND mcts_selfplay_test 42 10 40 4)
add_test(NAME threading COMMAND threading_test 8 100 256)
add_test(NAME selfplay_mt COMMAND selfplay_mt_test 4 4 40)
add_test(NAME arena COMMAND arena_test 12 4 30)
