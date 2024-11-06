cmake_minimum_required(VERSION 3.15)

project(t02284 LANGUAGES CXX)

if ($ENV{BUILD_DEBUG})
    add_definitions("-g -O0 -fno-strict-aliasing -fno-tree-vrp")
else()
    add_definitions("-O2")
endif()

set(CMAKE_CXX_STANDARD 23)


add_executable(t02284 main.cpp)


install(TARGETS t02284 DESTINATION bin)