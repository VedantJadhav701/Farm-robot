#!/usr/bin/env bash
# MI300X Pre-Flight Sanity & Run Layout Setup
# Usage: bash preflight.sh [session_name]
set -euo pipefail

SESSION_NAME="${1:-sessionX}"
export RUN="/scratch/runs/$(date +%Y%m%d)_${SESSION_NAME}"
mkdir -p "$RUN"/{data,out,logs}

echo "=== MI300X Pre-Flight ($SESSION_NAME) ==="
echo "RUN=$RUN"

rocm-smi || true
python3 -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', getattr(torch.version, 'hip', None), 'CUDA/ROCm avail:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Freeze environment for reproducibility
pip freeze > "$RUN/out/env.txt"
{
  echo "date=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "run=$RUN"
} > "$RUN/out/manifest.txt"

echo "Pre-flight complete. Remember: copy artifacts out before destroying the droplet!"
