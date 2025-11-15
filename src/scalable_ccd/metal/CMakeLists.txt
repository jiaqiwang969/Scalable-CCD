set(SOURCES
  broad_phase/broad_phase.hpp
  broad_phase/broad_phase.cpp
  broad_phase/aabb.hpp
  broad_phase/utils.hpp
  broad_phase/utils.cpp
  broad_phase/sweep.hpp
  broad_phase/sweep.cpp
  # Narrow-phase 代码已迁移为 legacy，暂不参与当前构建，避免影响验证流程
  # narrow_phase_legacy/*
  runtime/runtime.hpp
  runtime/runtime.mm
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(scalable_ccd PRIVATE ${SOURCES})

if(APPLE)
  # 链接 Apple Metal 框架
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(METALKIT_FRAMEWORK MetalKit)
  # MPS 可选
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  if(METAL_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${METAL_FRAMEWORK})
  endif()
  if(FOUNDATION_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${FOUNDATION_FRAMEWORK})
  endif()
  if(METALKIT_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${METALKIT_FRAMEWORK})
  endif()
  if(MPS_FRAMEWORK)
    target_link_libraries(scalable_ccd PRIVATE ${MPS_FRAMEWORK})
  endif()
  # 将 .mm 作为 ObjC++
  set_source_files_properties(
    narrow_phase/root_finder.mm
    narrow_phase/narrow_phase.mm
    runtime/runtime.mm
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-fobjc-arc -ObjC++"
  )
endif()
