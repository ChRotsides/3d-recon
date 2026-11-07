#!/bin/bash
# OpenVINS streaming runner. Built out of tree (GPL, desktop measurement only) following
# video-recon/vio/README.md; this wrapper only supplies the conda runtime it links against.
OV=${OV_ROOT:-$HOME/ai/vio}
export LD_LIBRARY_PATH="$OV/env/lib:$OV/open_vins/ov_msckf/build:$LD_LIBRARY_PATH"
exec "$OV/open_vins/ov_msckf/build/vio_stream" "$@"
