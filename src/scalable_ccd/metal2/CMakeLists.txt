if(APPLE AND SCALABLE_CCD_WITH_METAL2)
  target_sources(scalable_ccd PRIVATE
    runtime/runtime.mm
  )
  
  # Metal2 headers are already in the include path via root include dir
  
  # Link Metal framework
  find_library(METAL_LIBRARY Metal)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(QUARTZCORE_LIBRARY QuartzCore)
  
  target_link_libraries(scalable_ccd PRIVATE
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${QUARTZCORE_LIBRARY}
  )
  
  target_compile_definitions(scalable_ccd PUBLIC SCALABLE_CCD_WITH_METAL2)
endif()
