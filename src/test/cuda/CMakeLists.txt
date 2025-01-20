cmake_minimum_required(VERSION 3.20)
# 开启语言支持
project(cuda_demo LANGUAGES CXX CUDA)
if(CUDA_ENABLED)
    enable_language(CUDA)
endif()

#检查是否成功找到CUDA
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)
find_package(CUDA REQUIRED)
# 调用函数需要的头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{CUTLASS_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)
message(STATUS "LIBC_NAMESPACE: ${LIBC_NAMESPACE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{CUTLASS_DIR}/include \
    -I${CMAKE_CURRENT_SOURCE_DIR}/include \
    -I${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I$ENV{CUTLASS_DIR}/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/include \
    -I${CUDA_INCLUDE_DIRS}  -x cu")

add_executable(cuda main.cu)
# 链接编译后生成的库
target_link_libraries(cuda ${CUDA_LIBRARIES})

install(TARGETS cuda DESTINATION bin)
