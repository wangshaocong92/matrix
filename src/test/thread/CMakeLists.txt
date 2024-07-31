cmake_minimum_required(VERSION 3.15)

project(thread LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ./include
    ${Boost_INCLUDE_DIRS}
)

aux_source_directory(src SRC)

add_executable(thread main.cpp ${SRC})

target_link_libraries(thread Boost::boost pthread)


install(TARGETS thread DESTINATION bin)