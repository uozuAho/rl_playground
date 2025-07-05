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

function test_run() {
    pushd build
    ctest --output-on-failure
    popd
}

# build_full
build_fast

if [[ "$1" == "test" ]]; then
    test_run
else
    run
fi
