# ==============================================================================
# Tests
# ==============================================================================

add_executable(scalable_ccd_tests
  main.cpp
  test_broad_phase.cpp
  test_broad_phase_expected.cpp

  # Utils
  ground_truth.cpp
  ground_truth.hpp
  io.cpp
  io.hpp
)

if(APPLE AND SCALABLE_CCD_WITH_METAL)
  target_sources(scalable_ccd_tests PRIVATE
    test_broad_phase_metal.cpp
  )
  target_compile_definitions(scalable_ccd_tests PRIVATE SCALABLE_CCD_WITH_METAL)
endif()

if(SCALABLE_CCD_WITH_CUDA)
  target_sources(scalable_ccd_tests PRIVATE
    test_broad_phase.cu
    test_narrow_phase.cu
  )
endif()

# ==============================================================================
# Required Libraries
# ==============================================================================

target_link_libraries(scalable_ccd_tests PRIVATE scalable_ccd::scalable_ccd)

# Catch2
include(catch2)
target_link_libraries(scalable_ccd_tests PRIVATE Catch2::Catch2)

# libigl
include(libigl)
target_link_libraries(scalable_ccd_tests PRIVATE igl::core)

# JSON
include(json)
target_link_libraries(scalable_ccd_tests PRIVATE nlohmann_json::nlohmann_json)

# TBB
include(onetbb)
target_link_libraries(scalable_ccd_tests PRIVATE TBB::tbb)

# Sample data
include(sample_data)
target_link_libraries(scalable_ccd_tests PRIVATE scalable_ccd::data)

# Extra warnings (link last for highest priority)
include(scalable_ccd_warnings)
target_link_libraries(scalable_ccd_tests PRIVATE scalable_ccd::warnings)

# ==============================================================================
# Compiler options
# ==============================================================================

target_compile_definitions(scalable_ccd_tests PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)

# ==============================================================================
# Register tests
# ==============================================================================

FetchContent_GetProperties(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Register tests
set(PARSE_CATCH_TESTS_ADD_TO_CONFIGURE_DEPENDS ON)
catch_discover_tests(scalable_ccd_tests
  EXTRA_ARGS -s --durations yes
)
