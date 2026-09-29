cmake_minimum_required(VERSION 3.22.1)
project(edge_runtime LANGUAGES CXX)

# Keep the kernel registrar OBJECT files in the final .so, like the tinyqwen CLI.
get_filename_component(TINYQWEN_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../third_party/tiny-llm" ABSOLUTE)
set(TINYQWEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYQWEN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TINYQWEN_BUILD_CLI OFF CACHE BOOL "" FORCE)
# The app targets API 28+; use the same Vulkan DFlash graph as the CLI.
set(TINYQWEN_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
add_subdirectory("${TINYQWEN_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/tinyqwen")
if (NOT TINYQWEN_VULKAN_LIBRARY OR NOT TINYQWEN_GLSLC)
    message(FATAL_ERROR "Android DFlash requires Vulkan and glslc")
endif ()

add_library(edge_runtime SHARED jni_bridge.cpp)
target_compile_features(edge_runtime PRIVATE cxx_std_17)
target_link_libraries(edge_runtime PRIVATE tinyqwen_runtime tinyqwen_kernels log)
