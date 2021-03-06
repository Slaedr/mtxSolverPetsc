# test executables

add_executable(testbsrmatrix runbsrmatrixtest.cpp testbsrmatrix.cpp)
target_link_libraries(testbsrmatrix coomatrix blockmatrices)

add_executable(testcsrmatrix runcsrmatrixtest.cpp testcsrmatrix.cpp)
target_link_libraries(testcsrmatrix coomatrix blockmatrices)

add_executable(testcoomatrix runcoomatrixtest.cpp testcoomatrix.cpp testcsrmatrix.cpp)
target_link_libraries(testcoomatrix coomatrix blockmatrices)

# Tests

add_test(NAME COORead COMMAND ${SEQEXEC} ${SEQTASKS} testcoomatrix read
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted.coo)

add_test(NAME COOConvertToCSR
  COMMAND ${SEQEXEC} ${SEQTASKS} testcoomatrix convertCSR
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix_sorted.coo)

add_test(NAME COOConvertToBSR3
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcoomatrix convertBSR3
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted_bcolmajor.bcoo)

add_test(NAME CSRMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply matrix
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME CSRViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply view
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )

add_test(NAME BSR3ViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 3
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_b.mtx
  )

add_test(NAME BSR7MatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply matrix rowmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME BSR7ViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view rowmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME BSR7ColViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)

