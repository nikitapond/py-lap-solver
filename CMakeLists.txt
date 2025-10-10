cmake_minimum_required(VERSION 3.24)
project(py_lap_solver)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Optional: OpenMP support
# On macOS with Homebrew, help CMake find libomp
if(APPLE)
    # Try to find libomp from Homebrew
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include solver headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/batched_scipy)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/lap1015)

# Create the Python extension module for batched scipy solver
pybind11_add_module(_batched_scipy_lap
    src/cpp/bindings/batched_scipy_bindings.cpp
    src/cpp/batched_scipy/batched_scipy_lap.cpp
    src/cpp/batched_scipy/rectangular_lsap.cpp
)

# Create the Python extension module for LAP1015 solver
pybind11_add_module(_lap1015
    src/cpp/bindings/lap1015_bindings.cpp
)

# Compiler flags for optimization
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use -mcpu=native on ARM (Apple Silicon), -march=native on x86
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        target_compile_options(_batched_scipy_lap PRIVATE -O3 -mcpu=native -fstrict-aliasing)
        target_compile_options(_lap1015 PRIVATE -O3 -mcpu=native -fstrict-aliasing)
    else()
        target_compile_options(_batched_scipy_lap PRIVATE -O3 -march=native -fstrict-aliasing)
        target_compile_options(_lap1015 PRIVATE -O3 -march=native -fstrict-aliasing)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(_batched_scipy_lap PRIVATE /O2 /Wall)
    target_compile_options(_lap1015 PRIVATE /O2 /Wall)
endif()

# Enable OpenMP if available
if (OpenMP_CXX_FOUND)
    target_link_libraries(_batched_scipy_lap PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_batched_scipy_lap PRIVATE LAP_OPENMP)
    message(STATUS "Batched scipy solver: OpenMP support enabled")

    target_link_libraries(_lap1015 PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_lap1015 PRIVATE LAP_OPENMP)
    message(STATUS "LAP1015 solver: OpenMP support enabled")
else()
    message(STATUS "Batched scipy solver: OpenMP not found - parallel support disabled")
    message(STATUS "LAP1015 solver: OpenMP not found - parallel support disabled")
endif()

# Define LAP solver options
target_compile_definitions(_lap1015 PRIVATE LAP_QUIET)  # Suppress debug output

# Install the extension modules
install(TARGETS _batched_scipy_lap _lap1015 LIBRARY DESTINATION py_lap_solver)

# ============================================================================
# CUDA support (commented out - uncomment to enable when needed)
# ============================================================================
# include(CheckLanguage)
# check_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#     enable_language(CUDA)
#     if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#         set(CMAKE_CUDA_ARCHITECTURES auto)
#     endif()
#     target_compile_definitions(_lap1015 PRIVATE LAP_CUDA)
#     message(STATUS "LAP1015 solver: CUDA support enabled")
# else()
#     message(STATUS "LAP1015 solver: CUDA not found - GPU support disabled")
# endif()
