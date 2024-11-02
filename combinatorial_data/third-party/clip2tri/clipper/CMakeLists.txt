include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
  add_definitions("-DCLIPPER_DLL_EXPORTS")
endif(MSVC)

if(BUILD_SHARED_LIBS)
  add_library(clipper SHARED clipper.cpp)
  install(TARGETS clipper
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR})
endif(BUILD_SHARED_LIBS)

if(BUILD_STATIC_LIBS)
  add_library(clipper-static STATIC clipper.cpp)
  install(TARGETS clipper-static
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR})
endif(BUILD_STATIC_LIBS)

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8

