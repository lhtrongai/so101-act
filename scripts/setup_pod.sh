#!/usr/bin/env bash
# Set up a fresh Linux GPU pod (PyTorch image) for training with this repo.
# Usage: bash scripts/setup_pod.sh   (run from the repo root, then: source .venv/bin/activate)
set -euo pipefail

apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null   # faster video decoding for LeRobot
pip install -q uv
uv venv -q --python 3.12
source .venv/bin/activate
uv pip install -q -e ".[lerobot]"   # pins LeRobot to the commit used on the robot laptop

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| CUDA available:', torch.cuda.is_available())"
echo "CPU cores allotted to this pod may be fewer than nproc reports: $(nproc)"
echo "Next: hf auth login   (token with write access, or the browser login)"
