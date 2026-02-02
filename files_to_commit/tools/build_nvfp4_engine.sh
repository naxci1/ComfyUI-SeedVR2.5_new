#!/usr/bin/env bash
# Helper script for server: attempts to build TRT engine from ONNX using the nvfp4_engine builder
# Usage:
#   ./tools/build_nvfp4_engine.sh /path/to/model.onnx /path/to/output.plan
set -euo pipefail

ONNX_PATH="${1:-/tmp/seedvr2_nvfp4.onnx}"
PLAN_PATH="${2:-engines/nvfp4.plan}"
PYTHON="${PYTHON:-python3}"

echo "[nvfp4] ONNX: $ONNX_PATH"
echo "[nvfp4] PLAN: $PLAN_PATH"

$PYTHON - <<PYCODE
from src.runtime.tensorrt.nvfp4_engine import build_engine_from_onnx, is_trt_available, get_trt_version
import sys, os
print("TensorRT available:", is_trt_available())
try:
    print("TensorRT version:", get_trt_version())
except Exception:
    pass
build_engine_from_onnx("$ONNX_PATH", "$PLAN_PATH", enable_nvfp4=True)
print("Engine build completed:", "$PLAN_PATH")
PYCODE
