cmake_minimum_required(VERSION 3.15)

project(sum_matrix LANGUAGES CXX)

if ($ENV{BUILD_DEBUG})
    add_definitions("-g -O0 -fno-strict-aliasing -fno-tree-vrp")
else()
    add_definitions("-O2")
endif()

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(sum_matrix sum_matrix.cpp)


install(TARGETS sum_matrix DESTINATION bin)