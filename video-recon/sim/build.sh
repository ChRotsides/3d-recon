#!/bin/bash
# Builds the ctypes shim around the phone's voxel-hash TSDF. tsdf.cpp is compiled unchanged.
set -e
d="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$d/build"
g++ -O2 -std=c++17 -shared -fPIC -o "$d/build/libtsdf.so" \
    "$d/tsdf_c.cpp" "$d/../android/app/src/main/cpp/tsdf.cpp"
echo "built $d/build/libtsdf.so"
