#!/bin/bash

set -eu

LIBTORCH_URL=https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
SRC_DIR=$(pwd)/src
LIBTORCH_DIR=${SRC_DIR}/libtorch

BUILD_TYPE=Debug
BUILD_FULL=false
BUILD_TESTS=OFF
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--rebuild" ]]; then
        BUILD_FULL=true
    elif [[ "$arg" == "--release" ]]; then
        BUILD_TYPE=Release
    elif [[ "$arg" == "--debug" ]]; then
        BUILD_TYPE=Debug
    elif [[ "$arg" == "test" ]]; then
        BUILD_TESTS=ON
        ARGS+=("$arg")
    else
        ARGS+=("$arg")
    fi
done

function fetch_torch() {
    if [ ! -d "$LIBTORCH_DIR" ]; then
        mkdir -p /tmp/libtorch
        pushd /tmp/libtorch
        wget $LIBTORCH_URL
        unzip libtorch-shared-with-deps-latest.zip -d $SRC_DIR
        popd
    fi
}

function build_full() {
    rm -rf build
    fetch_torch
    mkdir build
    pushd build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DTESTS_ENABLED=$BUILD_TESTS ..
    make
    popd
}

function build_fast() {
    fetch_torch
    pushd build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DTESTS_ENABLED=$BUILD_TESTS ..
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

if $BUILD_FULL; then
    build_full
else
    build_fast
fi

if [[ ${#ARGS[@]} -gt 0 && "${ARGS[0]}" == "test" ]]; then
    test_run
else
    run
fi
