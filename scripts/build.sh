#!/bin/bash
env=$(cd "$(dirname "$0")"; pwd)/env.sh
source $env
cpu=$(cat /proc/cpuinfo | grep "processor" | wc -l)
export BUILD_DEBUG=ON

rm -rf install || true
rm -rf build || true
# conan
mkdir build 
cd build
conan install ..  --build=missing -o with_cuda=11.8.0 -o with_tensorrt=8.5.3.1  -o with_opencv_gpu=True 
cmake $PROJECT_PATH -DCMAKE_INSTALL_PREFIX=$WORKSPACE_PATH/install  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

make -j$cpu
make install

cp compile_commands.json ../../
