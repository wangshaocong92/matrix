cmake_minimum_required(VERSION 3.15)


if ($ENV{BUILD_DEBUG})  
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

# add_compile_definitions(-Wall -Wextra -pedantic -Werror)

add_subdirectory(src)