add_subdirectory(utils)
add_subdirectory(kernels/gemm)
add_subdirectory(api/python)

add_executable(test test.cc)
target_link_libraries(test PUBLIC _turbomind_ext_static)
target_link_libraries(test PRIVATE gemm2)
