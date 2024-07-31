cmake_minimum_required(VERSION 3.15)

project(gcc14 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    include
    ${Boost_INCLUDE_DIRS}
)

aux_source_directory(src SRC)

add_executable(gcc14 main.cpp ${SRC})

target_link_libraries(gcc14 Boost::boost pthread)