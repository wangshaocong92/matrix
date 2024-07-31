cmake_minimum_required(VERSION 3.20)
# 开启语言支持
project(matrix_demo LANGUAGES CXX CUDA)
if(CUDA_ENABLED)
    enable_language(CUDA)
endif()
#检查是否成功找到CUDA
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDA)

# 调用函数需要的头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
message(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_library(matrix  SHARED ${CUDA_SRC})

add_executable(main main.cu)
# 链接编译后生成的库
target_link_libraries(main matrix)
