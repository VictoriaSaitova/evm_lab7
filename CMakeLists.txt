cmake_minimum_required(VERSION 3.20)
project(evm_lab7 C)

set(CMAKE_C_STANDARD 11)

# Три варианта программы
add_executable(plain plain.c)   # без ручной векторизации
add_executable(sse   sse.c)     # SSE intrinsics
add_executable(blas  blas.c)    # BLAS

foreach(target plain sse blas)
    target_compile_options(${target} PRIVATE -O3)
    target_link_libraries(${target} m)
endforeach()

# OpenBLAS (в MSYS2 заголовок cblas.h лежит в include/openblas)
find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
target_include_directories(blas PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(blas openblas)
