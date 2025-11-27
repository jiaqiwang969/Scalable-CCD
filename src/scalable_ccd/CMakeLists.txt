set(SOURCES
  config.hpp
  scalar.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

# ==============================================================================
# Subfolders
# ==============================================================================
add_subdirectory(broad_phase)
add_subdirectory(utils)

if(SCALABLE_CCD_WITH_CUDA)
  add_subdirectory(cuda)
endif()

if(SCALABLE_CCD_WITH_METAL2)
  add_subdirectory(metal2)
endif()