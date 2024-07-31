export CC=/usr/local/gcc-14.1.0/bin/gcc
export CXX=/usr/local/gcc-14.1.0/bin/g++

export LD_LIBRARY_PATH=/usr/local/gcc-14.1.0/lib:/usr/local/gcc-14.1.0/lib64:$LD_LIBRARY_PATH
 
export WORKSPACE_PATH="$(cd "$(dirname "$0")"; pwd)/../"
export PROJECT_PATH="$(cd "$(dirname "$0")"; pwd)/../../"
