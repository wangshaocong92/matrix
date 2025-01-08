cmake_minimum_required(VERSION 3.15)

project(openmp_test LANGUAGES CXX)

if ($ENV{BUILD_DEBUG})
    add_definitions("-g -O0 -fno-strict-aliasing -fno-tree-vrp")
else()
    add_definitions("-O2")
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 23)

add_executable(parallel parallel.cpp)
target_link_libraries(parallel OpenMP::OpenMP_CXX)
install(TARGETS parallel DESTINATION bin/openmp_test)

add_executable(parallel_for parallel_for.cpp)
target_link_libraries(parallel_for OpenMP::OpenMP_CXX)
install(TARGETS parallel_for DESTINATION bin/openmp_test)

add_executable(parallel_section parallel_section.cpp)
target_link_libraries(parallel_section OpenMP::OpenMP_CXX)
install(TARGETS parallel_section DESTINATION bin/openmp_test)

add_executable(parallel_signal_master parallel_signal_master.cpp)
target_link_libraries(parallel_signal_master OpenMP::OpenMP_CXX)
install(TARGETS parallel_signal_master DESTINATION bin/openmp_test)

add_executable(barriar barriar.cpp)
target_link_libraries(barriar OpenMP::OpenMP_CXX)
install(TARGETS barriar DESTINATION bin/openmp_test)

add_executable(atomic atomic.cpp)
target_link_libraries(atomic OpenMP::OpenMP_CXX)
install(TARGETS atomic DESTINATION bin/openmp_test)

add_executable(critical critical.cpp)
target_link_libraries(critical OpenMP::OpenMP_CXX)
install(TARGETS critical DESTINATION bin/openmp_test)

add_executable(threadprivate threadprivate.cpp)
target_link_libraries(threadprivate OpenMP::OpenMP_CXX)
install(TARGETS threadprivate DESTINATION bin/openmp_test)

add_executable(if if.cpp)
target_link_libraries(if OpenMP::OpenMP_CXX)
install(TARGETS if DESTINATION bin/openmp_test)


add_executable(private private.cpp)
target_link_libraries(private OpenMP::OpenMP_CXX)
install(TARGETS private DESTINATION bin/openmp_test)

add_executable(shared shared.cpp)
target_link_libraries(shared OpenMP::OpenMP_CXX)
install(TARGETS shared DESTINATION bin/openmp_test)

add_executable(reduction reduction.cpp)
target_link_libraries(reduction OpenMP::OpenMP_CXX)
install(TARGETS reduction DESTINATION bin/openmp_test)