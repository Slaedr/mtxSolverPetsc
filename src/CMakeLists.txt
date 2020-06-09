# For some reason, CMake can't figure out PIC flag for Cray
if(CXX_COMPILER_CRAY)
  add_compile_options(-fPIC)
endif()

add_subdirectory(blas)

add_library(helper helper_algorithms.cpp)
#set_property(TARGET helper PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(rawmatrixutils rawsrmatrixutils.cpp)
#set_property(TARGET rawmatrixutils PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(blockmatrices csrmatrix.cpp bsrmatrix.cpp)
#set_property(TARGET blockmatrices PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blockmatrices myblas rawmatrixutils)

add_library(coomatrix coomatrix.cpp)
target_link_libraries(coomatrix blockmatrices)
#set_property(TARGET coomatrix PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(utils cmdoptions.cpp)
#target_link_libraries(utils ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_library(petsctestutils testutils.cpp testutils.c)
target_link_libraries(petsctestutils rawmatrixutils utils)
#set_property(TARGET petsctestutils PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(petscsingle runpetscsingle.c)
target_link_libraries(petscsingle coomatrix petsctestutils
  ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_executable(testpetscsolver runpetsctest.c)
target_link_libraries(testpetscsolver coomatrix petsctestutils
  ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

add_executable(util_petsc_io util_petsc_io.c blasted_petsc_io.cpp)
target_link_libraries(util_petsc_io coomatrix blockmatrices
  ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

