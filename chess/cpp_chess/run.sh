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

BUILD_FULL=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--rebuild" ]]; then
        BUILD_FULL=true
    else
        ARGS+=("$arg")
    fi
done

if $BUILD_FULL; then
    build_full
else
    build_fast
fi

if [[ "${ARGS[0]}" == "test" ]]; then
    test_run
else
    run
fi
