#!/usr/bin/env bash
# 在新机器上建立与开发机一致的 Python 环境（项目根目录下的 .venv）。
#
#   bash scripts/setup_env.sh                  # 默认 CUDA 12.6 版 torch（驱动 ≥ 525 即可）
#   TORCH_VARIANT=cpu bash scripts/setup_env.sh
#
# 需要 uv（https://docs.astral.sh/uv/）；没有 uv 时退回 python3 -m venv + pip。
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
TORCH_VARIANT="${TORCH_VARIANT:-cu126}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_VARIANT}"

if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_VERSION}" .venv
    uv pip install --python .venv/bin/python -r requirements-lock.txt
    uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
else
    python3 -m venv .venv
    .venv/bin/pip install -r requirements-lock.txt
    .venv/bin/pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
fi

.venv/bin/python - <<'EOF'
import sys, numpy, scipy, matplotlib, gymnasium, torch
print("python", sys.version.split()[0])
for m in (numpy, scipy, matplotlib, gymnasium, torch):
    print(f"{m.__name__:11s}", m.__version__)
print("cuda available:", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
