#!/bin/bash
set -e
echo "=== ROCm GPU Check ==="
rocm-smi --showid 2>/dev/null | head -8 || echo "rocm-smi not in PATH"
echo ""
echo "=== PyTorch + HIP ==="
python3 << 'PYEOF'
import torch
print("PyTorch:", torch.__version__)
print("HIP:", getattr(torch.version, "hip", "N/A"))
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM:", round(props.total_memory / 1e9, 1), "GB")
PYEOF
echo ""
echo "=== Disk ==="
df -h /scratch 2>/dev/null || df -h /
echo ""
echo "=== Installed ML packages ==="
pip list 2>/dev/null | grep -iE "torch|transformers|peft|trl|datasets|vllm|accelerate|bitsandbytes|sentencepiece" || echo "none found"
echo "=== DONE ==="
