#!/bin/bash

set -eu

BUILD_TYPE=Debug
BUILD_FULL=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--rebuild" ]]; then
        BUILD_FULL=true
    elif [[ "$arg" == "--release" ]]; then
        BUILD_TYPE=Release
    elif [[ "$arg" == "--debug" ]]; then
        BUILD_TYPE=Debug
    else
        ARGS+=("$arg")
    fi
done

function build_full() {
    rm -rf build
    mkdir build
    pushd build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    make
    popd
}

function build_fast() {
    pushd build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
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
