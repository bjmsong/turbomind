#!/bin/bash
WORKSPACE_PATH=$(dirname "$(readlink -f "$0")")

builder="-G Ninja"

if [ "$1" == "make" ]; then
    builder=""
fi

cmake ${builder} .. \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_INSTALL_PREFIX=${WORKSPACE_PATH}/install \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON \
    -DBUILD_TEST=ON
