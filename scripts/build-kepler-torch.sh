#!/usr/bin/env bash
# Build the repo's CUDA 11.8 / sm_35 PyTorch wheel without consuming disk for
# the CUDA SDK or source tree. The persistent result is .venv-kepler.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_prefix=${KEPLER_CUDA_PREFIX:-/dev/shm/cuda-11.8}
torch_src=${KEPLER_TORCH_SRC:-/dev/shm/pytorch-kepler}
venv_prefix=${KEPLER_VENV:-"$repo_root/.venv-kepler"}
build_jobs=${MAX_JOBS:-6}
conda_exe=${KEPLER_CONDA_EXE:-$(command -v conda || true)}
uv_exe=${KEPLER_UV_EXE:-$(command -v uv || true)}

if [[ -z "$conda_exe" || -z "$uv_exe" ]]; then
    echo "conda and uv must be on PATH (or set KEPLER_CONDA_EXE/KEPLER_UV_EXE)" >&2
    exit 2
fi

cudnn_prefix=${KEPLER_CUDNN_PREFIX:-$($conda_exe run -n kepler python -c 'import sys; print(sys.prefix)')}
if [[ ! -f "$cudnn_prefix/include/cudnn.h" ]]; then
    echo "cuDNN headers not found in $cudnn_prefix" >&2
    exit 2
fi

# Keep the existing Conda environment as the small, persistent CUDA runtime.
# The compiler/source prefixes below live in tmpfs and disappear on reboot.
if ! "$conda_exe" list -p "$cudnn_prefix" cudatoolkit | grep -q '^cudatoolkit[[:space:]]*11\.8'; then
    "$conda_exe" install -y -p "$cudnn_prefix" -c conda-forge \
        --override-channels cudatoolkit=11.8 cuda-version=11.8
fi

if [[ ! -x "$cuda_prefix/bin/nvcc" ]]; then
    "$conda_exe" create -y -p "$cuda_prefix" --override-channels \
        -c nvidia/label/cuda-11.8.0 -c conda-forge \
        cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-driver-dev=11.8 \
        cuda-nvrtc-dev=11.8 cuda-nvtx=11.8 cuda-nvml-dev=11.8 \
        cuda-profiler-api=11.8 \
        libcublas-dev=11.11 libcufft-dev=10.9 libcurand-dev=10.3 \
        libcusolver-dev=11.4 libcusparse-dev=11.7 \
        gcc_linux-64=11 gxx_linux-64=11
fi

if [[ ! -f "$cuda_prefix/include/cuda_profiler_api.h" ]]; then
    "$conda_exe" install -y -p "$cuda_prefix" --override-channels \
        -c nvidia cuda-profiler-api=11.8
fi

# PyTorch's finder expects cuDNN inside a CUDA-like prefix. Symlinks keep the
# 8.9 runtime in the existing Conda environment and avoid another multi-GB copy.
for header in "$cudnn_prefix"/include/cudnn*.h; do
    ln -sfn "$header" "$cuda_prefix/include/$(basename "$header")"
done
ln -sfn "$cudnn_prefix/lib/libcudnn.so" "$cuda_prefix/lib/libcudnn.so"
ln -sfn "$cudnn_prefix/lib/libcudnn.so.8" "$cuda_prefix/lib/libcudnn.so.8"

# Conda's GCC sysroot does not know Ubuntu's multiarch Python include location.
# Python.h includes this directory by name when building torch_python.
ln -sfn /usr/include/x86_64-linux-gnu \
    "$cuda_prefix/x86_64-conda-linux-gnu/sysroot/usr/include/x86_64-linux-gnu"

if [[ ! -x "$venv_prefix/bin/python" ]]; then
    "$uv_exe" venv "$venv_prefix" --python 3.12
fi
"$uv_exe" pip install --python "$venv_prefix/bin/python" \
    'numpy<2' setuptools wheel pyyaml typing-extensions sympy filelock \
    networkx jinja2 fsspec requests packaging build

if [[ ! -d "$torch_src/.git" ]]; then
    git clone --recursive --shallow-submodules --depth 1 --branch v2.7.1 \
        https://github.com/pytorch/pytorch.git "$torch_src"
fi
"$uv_exe" pip install --python "$venv_prefix/bin/python" \
    -r "$torch_src/requirements.txt"

cd "$torch_src"
env \
    PATH="$cuda_prefix/bin:$venv_prefix/bin:/usr/bin:/bin" \
    CC="$cuda_prefix/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="$cuda_prefix/bin/x86_64-conda-linux-gnu-g++" \
    CUDAHOSTCXX="$cuda_prefix/bin/x86_64-conda-linux-gnu-g++" \
    CUDA_HOME="$cuda_prefix" \
    CUDNN_ROOT="$cuda_prefix" \
    CUDNN_INCLUDE_DIR="$cuda_prefix/include" \
    CUDNN_LIBRARY="$cuda_prefix/lib/libcudnn.so" \
    CMAKE_PREFIX_PATH="$venv_prefix:$cuda_prefix" \
    LD_LIBRARY_PATH="$cuda_prefix/lib" \
    TORCH_CUDA_ARCH_LIST=3.5 \
    MAX_JOBS="$build_jobs" \
    USE_CUDA=1 USE_CUDNN=1 USE_CUFILE=0 USE_CUSPARSELT=0 \
    USE_KINETO=0 USE_CUPTI=0 USE_DISTRIBUTED=0 USE_NCCL=0 \
    USE_GLOO=0 USE_TENSORPIPE=0 USE_MPI=0 USE_XPU=0 \
    USE_MKLDNN=0 USE_FBGEMM=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
    USE_XNNPACK=0 USE_NNPACK=0 USE_MIMALLOC=0 USE_ITT=0 \
    USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_OPENMP=1 \
    BUILD_TEST=0 BUILD_BINARY=0 BUILD_FUNCTORCH=0 \
    PYTORCH_BUILD_VERSION=2.7.1+cu118.kepler PYTORCH_BUILD_NUMBER=1 \
    TORCH_NVCC_FLAGS='-Xfatbin -compress-all' \
    "$venv_prefix/bin/python" setup.py bdist_wheel

wheel_path=$(find "$torch_src/dist" -maxdepth 1 -name 'torch-*.whl' -print -quit)
if [[ -z "$wheel_path" ]]; then
    echo "PyTorch build completed without producing a wheel" >&2
    exit 1
fi
"$uv_exe" pip install --python "$venv_prefix/bin/python" --no-deps "$wheel_path"

# The source-built wheel intentionally does not duplicate NVIDIA's runtime.
# Put links beside libtorch, whose $ORIGIN rpath the loader already searches.
torch_lib="$venv_prefix/lib/python3.12/site-packages/torch/lib"
for library in \
    "$cudnn_prefix"/lib/libcudart.so* \
    "$cudnn_prefix"/lib/libcublas.so* \
    "$cudnn_prefix"/lib/libcublasLt.so* \
    "$cudnn_prefix"/lib/libcudnn*.so* \
    "$cudnn_prefix"/lib/libcufft.so* \
    "$cudnn_prefix"/lib/libcurand.so* \
    "$cudnn_prefix"/lib/libcusolver.so* \
    "$cudnn_prefix"/lib/libcusparse.so* \
    "$cudnn_prefix"/lib/libnvrtc.so* \
    "$cudnn_prefix"/lib/libnvrtc-builtins.so*; do
    [[ -e "$library" ]] || continue
    ln -sfn "$library" "$torch_lib/$(basename "$library")"
done

# Keep torchvision CPU-only: its image transforms are useful to the models,
# while its stock CUDA wheel would pull in a newer CUDA runtime and undo this
# environment. DA3's API imports several export helpers eagerly, so the small
# collection below includes their import-time dependencies even though depth
# inference does not call those exporters. xFormers/gsplat remain deliberately
# absent because their released CUDA extensions do not contain sm_35 kernels.
"$uv_exe" pip install --python "$venv_prefix/bin/python" --no-deps \
    'torchvision==0.22.1+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
"$uv_exe" pip install --python "$venv_prefix/bin/python" -e "$repo_root"
"$uv_exe" pip install --python "$venv_prefix/bin/python" \
    'numpy==1.26.4' 'opencv-python-headless==4.10.0.84' \
    addict e3nn einops evo 'huggingface-hub>=0.34' imageio matplotlib \
    'moviepy==1.0.3' omegaconf pillow-heif plyfile pycolmap requests \
    safetensors 'transformers==5.0.0'
"$uv_exe" pip install --python "$venv_prefix/bin/python" --no-deps \
    'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
echo "Installed $wheel_path into $venv_prefix"
