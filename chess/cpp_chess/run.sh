#!/bin/bash

function build_full() {
    rm -rf build
    mkdir build
    pushd build
    cmake ..
    make
    popd
}

function build_fast() {
    pushd build
    make
    popd
}

function run() {
    pushd build
    ./cpp_chess
    popd
}

# build_full
build_fast
# run
