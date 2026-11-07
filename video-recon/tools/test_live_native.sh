#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${LIVE_HOST_PREFIX:-"$project_dir/.deps/native-host"}
build=${LIVE_HOST_BUILD:-"$project_dir/.deps/host-build"}
sdk=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-"$HOME/Android/Sdk"}}
ninja_path=${LIVE_NINJA:-"$sdk/cmake/3.22.1/bin/ninja"}
if [[ ! -f "$prefix/lib/cmake/Ceres/CeresConfig.cmake" ]]; then
    echo "Missing native host dependencies in $prefix. See docs/live-reliability-2026-10-07.md." >&2
    exit 1
fi
cmake -S "$project_dir/android/app/src/main/cpp" -B "$build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja_path" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$prefix" -DTHIRD_PARTY_INCLUDE="$prefix/include" \
    -DCeres_DIR="$prefix/lib/cmake/Ceres" -DOpenCV_DIR="$prefix/lib/cmake/opencv4" \
    -DEigen3_DIR="$prefix/share/eigen3/cmake"
cmake --build "$build" --parallel "${LIVE_BUILD_JOBS:-2}" --target \
    feature_tracking_test handheld_init_test map_relocalizer_test stationary_test tsdf_quality_test
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir "$build" --output-on-failure
config="$project_dir/android/app/src/main/assets/vio/estimator_config.yaml"
"$build/feature_tracking_test" "$config" 31 "$build/validated-surface.ply"
"$build/handheld_init_test" "$config"
