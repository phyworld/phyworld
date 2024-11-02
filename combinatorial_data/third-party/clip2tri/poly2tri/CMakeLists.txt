include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(poly2tri_srcs
  common/shapes.cc
  sweep/advancing_front.cc
  sweep/cdt.cc
  sweep/sweep.cc
  sweep/sweep_context.cc
)

if(MSVC)
  add_definitions("-DP2T_DLL_EXPORTS")
endif(MSVC)

if(BUILD_SHARED_LIBS)
  add_library(poly2tri SHARED ${poly2tri_srcs})
  target_link_libraries(poly2tri ${M_LIBRARY})
  install(TARGETS poly2tri
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR})
endif(BUILD_SHARED_LIBS)

if(BUILD_STATIC_LIBS)
  add_library(poly2tri-static STATIC ${poly2tri_srcs})
  target_link_libraries(poly2tri-static ${M_LIBRARY})
  install(TARGETS poly2tri-static
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

