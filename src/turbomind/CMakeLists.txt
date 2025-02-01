add_subdirectory(utils)
add_subdirectory(kernels/gemm)
add_subdirectory(api/python)

add_executable(test test_linear.cc)
target_link_libraries(test PUBLIC _turbomind_ext_static)
target_link_libraries(test PRIVATE gemm2)
find_package(GTest REQUIRED)
target_link_libraries(test
    PRIVATE
    GTest::GTest
    GTest::Main
    pthread
)