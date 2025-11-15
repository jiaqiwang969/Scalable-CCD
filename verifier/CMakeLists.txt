# Verifier executable: run scenarios, collect env info, measure timings, and emit JSON/HTML reports.

cmake_minimum_required(VERSION 3.18)

# The verifier lives inside this project; it uses the built library and recipes.

add_executable(scalable_ccd_verifier
  main.cpp
  env_info.cpp
  env_info.hpp
  io.cpp
  io.hpp
  compare.cpp
  compare.hpp
  report.cpp
  report.hpp
  queries.cpp
  queries.hpp
)

# Public include to access scalable_ccd headers
target_include_directories(scalable_ccd_verifier PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Required libs
target_link_libraries(scalable_ccd_verifier PRIVATE
  scalable_ccd::scalable_ccd
)

# Logger
include(spdlog)
target_link_libraries(scalable_ccd_verifier PRIVATE spdlog::spdlog)

# JSON
include(json)
target_link_libraries(scalable_ccd_verifier PRIVATE nlohmann_json::nlohmann_json)

# libigl for mesh IO (PLY/OBJ)
include(libigl)
target_link_libraries(scalable_ccd_verifier PRIVATE igl::core)

# TBB for CPU build_* helpers
include(onetbb)
target_link_libraries(scalable_ccd_verifier PRIVATE TBB::tbb)

# Sample data path define and fetch (reuse if tests already added it)
if(TARGET scalable_ccd::data)
  target_link_libraries(scalable_ccd_verifier PRIVATE scalable_ccd::data)
else()
  include(sample_data)
  target_link_libraries(scalable_ccd_verifier PRIVATE scalable_ccd::data)
endif()

# Extra warnings (align with project)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/scalable_ccd/scalable_ccd_warnings.cmake)
target_link_libraries(scalable_ccd_verifier PRIVATE scalable_ccd::warnings)

# Export build config to code
target_compile_definitions(scalable_ccd_verifier PRIVATE
  SCALABLE_CCD_BUILD_TYPE="$<IF:$<CONFIG:Debug>,Debug,Release>"
)

add_executable(scalable_ccd_aggregate
  aggregate.cpp
  report.cpp
  report.hpp
)
target_link_libraries(scalable_ccd_aggregate PRIVATE
  nlohmann_json::nlohmann_json
  spdlog::spdlog
)
