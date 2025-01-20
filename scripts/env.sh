export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
 
export CUTLASS_DIR=/usr/local/cutlass

export WORKSPACE_PATH="$(cd "$(dirname "$0")"; pwd)/../"
export PROJECT_PATH=$WORKSPACE_PATH/../
export PYTHONPATH=$WORKSPACE_PATH/src:$PYTHONPATH
cd $WORKSPACE_PATH
export LD_LIBRARY_PATH=$(pwd)/install/lib:$LD_LIBRARY_PATH