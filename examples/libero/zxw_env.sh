# Source this file before installing, training, or evaluating pi0.5 on LIBERO.
# It keeps downloaded models, datasets, package caches, logs, and generated
# artifacts in zxw's personal storage instead of system-wide or root-owned paths.

export OPENPI_PROJECT_ROOT="/home/pai/zxw/openpi"
export PI05_LIBERO_ROOT="/home/pai/zxw/openpi_data/pi05_libero"

export XDG_CACHE_HOME="${PI05_LIBERO_ROOT}/cache/xdg"
export OPENPI_DATA_HOME="${PI05_LIBERO_ROOT}/cache/openpi"
export HF_HOME="${PI05_LIBERO_ROOT}/cache/huggingface"
# The node cannot reach huggingface.co directly. This domestic endpoint has
# been verified against both the LIBERO refs API and ranged parquet downloads.
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
export HF_XET_CACHE="${HF_HOME}/xet"
# The LIBERO repository is Xet-backed. Use concurrent ranged downloads through
# the proxy inherited from the current shell and tolerate long large-file reads.
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=16
export HF_HUB_DOWNLOAD_TIMEOUT=600
# Hugging Face uses a separate timeout for HEAD/metadata requests. The mirror
# occasionally needs more than the 10-second default when resolving 1699 files.
export HF_HUB_ETAG_TIMEOUT=600
# Clear the legacy variable if an older version of this script exported it.
unset TRANSFORMERS_CACHE
export TORCH_HOME="${PI05_LIBERO_ROOT}/cache/torch"
export UV_CACHE_DIR="${PI05_LIBERO_ROOT}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${PI05_LIBERO_ROOT}/python"
# UV_DEFAULT_INDEX is the current uv spelling. Keep UV_INDEX_URL for tools and
# older uv releases that still consume the pip-compatible variable.
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export UV_INDEX_URL="${UV_DEFAULT_INDEX}"
export PIP_CACHE_DIR="${PI05_LIBERO_ROOT}/cache/pip"
export JAX_COMPILATION_CACHE_DIR="${PI05_LIBERO_ROOT}/cache/jax"
export CUDA_CACHE_PATH="${PI05_LIBERO_ROOT}/cache/cuda"
export TRITON_CACHE_DIR="${PI05_LIBERO_ROOT}/cache/triton"
export TORCH_EXTENSIONS_DIR="${PI05_LIBERO_ROOT}/cache/torch_extensions"
export WANDB_DIR="${PI05_LIBERO_ROOT}/wandb"
export WANDB_CACHE_DIR="${PI05_LIBERO_ROOT}/cache/wandb"
export WANDB_ARTIFACT_DIR="${PI05_LIBERO_ROOT}/wandb/artifacts"
export TMPDIR="${PI05_LIBERO_ROOT}/tmp"
export LIBERO_CONFIG_PATH="${OPENPI_PROJECT_ROOT}/examples/libero/zxw_config"

mkdir -p \
    "${OPENPI_DATA_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_ASSETS_CACHE}" \
    "${HF_LEROBOT_HOME}" \
    "${HF_XET_CACHE}" \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_ARTIFACT_DIR}" \
    "${TMPDIR}" \
    "${PI05_LIBERO_ROOT}/assets" \
    "${PI05_LIBERO_ROOT}/checkpoints" \
    "${PI05_LIBERO_ROOT}/logs" \
    "${PI05_LIBERO_ROOT}/videos" \
    "${PI05_LIBERO_ROOT}/venvs"
