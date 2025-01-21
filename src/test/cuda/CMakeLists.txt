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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{CUTLASS_DIR}/include \
    -I${CMAKE_CURRENT_SOURCE_DIR}/include \
    -I${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I$ENV{CUTLASS_DIR}/include  \
    -I${CMAKE_CURRENT_SOURCE_DIR}/include \
    -I${CUDA_INCLUDE_DIRS}  -x cu")

# set(CUDA_SOURCES src/matmul.cu)
# add_library(cuda_lib SHARED ${CUDA_SOURCES})
# target_compile_options(cuda_lib PRIVATE -G)  # 启用调试信息
# target_link_libraries(cuda_lib ${CUDA_LIBRARIES})
add_executable(cuda main.cu src/matmul.cu)
# 链接编译后生成的库
target_link_libraries(cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})


install(TARGETS cuda DESTINATION bin)
# install(TARGETS cuda_lib DESTINATION lib)
