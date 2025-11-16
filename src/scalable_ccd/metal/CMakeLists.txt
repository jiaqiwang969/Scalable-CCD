cmake_minimum_required(VERSION 3.18)

# Build Metal kernels and host wrapper, link frameworks, and expose to main target.

if(NOT APPLE)
  message(WARNING "Metal backend requested but not on Apple platform; skipping.")
  return()
endif()

set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/broad_phase_sap.metal)
set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/broad_phase_sap.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/scalable_ccd_kernels.metallib)

add_custom_command(
  OUTPUT ${METAL_AIR}
  COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
  DEPENDS ${METAL_SRC}
  VERBATIM
  COMMENT "Compiling Metal kernel to AIR: ${METAL_SRC}"
)

add_custom_command(
  OUTPUT ${METAL_LIB}
  COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
  DEPENDS ${METAL_AIR}
  VERBATIM
  COMMENT "Linking Metal AIR to metallib: ${METAL_LIB}"
)

add_custom_target(scalable_ccd_metallib ALL DEPENDS ${METAL_LIB})

# Host wrapper
set(METAL_HOST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/broad_phase_metal.cpp
)

target_sources(scalable_ccd PRIVATE ${METAL_HOST_SOURCES})

target_include_directories(scalable_ccd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/scalable_ccd/metal-cpp
)

target_compile_definitions(scalable_ccd PRIVATE
  SCALABLE_CCD_METAL_LIB_PATH="${METAL_LIB}"
)

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(QUARTZCORE_FRAMEWORK QuartzCore)

target_link_libraries(scalable_ccd PRIVATE
  ${FOUNDATION_FRAMEWORK}
  ${METAL_FRAMEWORK}
  ${QUARTZCORE_FRAMEWORK}
)
