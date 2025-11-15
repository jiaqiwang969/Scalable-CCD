set(SOURCES
  broad_phase/broad_phase.hpp
  broad_phase/broad_phase.cpp
  broad_phase/aabb.hpp
  runtime/runtime.hpp
  runtime/runtime.mm
  runtime/metalcpp_symbols.mm
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

if(APPLE)
  # metal-cpp headers
  target_include_directories(scalable_ccd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../metal-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../metal-cpp/Metal
    ${CMAKE_CURRENT_SOURCE_DIR}/../metal-cpp/Foundation
  )
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${METAL_FRAMEWORK})
  endif()
  if(FOUNDATION_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${FOUNDATION_FRAMEWORK})
  endif()
  set_source_files_properties(
    runtime/metalcpp_symbols.mm
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-fobjc-arc -ObjC++"
  )
  set_source_files_properties(
    runtime/runtime.mm
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-fobjc-arc -ObjC++"
  )
endif()
