#!/bin/bash

# build: full
# rm -rf build
# mkdir build
# pushd build
# cmake ..
# make

# build: fast
pushd build
make

# run
./cpp_chess

popd
