#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
if [[ "${machine}" != riscv* && "${ALLOW_NON_RISCV:-0}" != "1" ]]; then
  echo "This smoke test must run on a RISC-V host. Detected: ${machine}" >&2
  echo "Set ALLOW_NON_RISCV=1 only for Python syntax/debug checks." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

detect_vlen() {
  if [[ -n "${VLLM_RVV_VLEN:-}" ]]; then
    echo "${VLLM_RVV_VLEN}"
    return
  fi
  if [[ -r /proc/cpuinfo ]]; then
    local best
    best="$(
      grep -Eo 'zvl[0-9]+b' /proc/cpuinfo \
        | sed -E 's/zvl([0-9]+)b/\1/' \
        | sort -n \
        | tail -n 1
    )"
    if [[ -n "${best}" ]]; then
      echo "${best}"
      return
    fi
  fi
  echo "128"
}

export VLLM_TARGET_DEVICE=cpu
export VLLM_RVV_VLEN="$(detect_vlen)"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "Using VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}"
echo "Using VLLM_RVV_VLEN=${VLLM_RVV_VLEN}"
echo "Using MAX_JOBS=${MAX_JOBS}"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "${PYTHON_VERSION:-3.12}"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
  VLLM_RVV_VLEN="${VLLM_RVV_VLEN}" \
  uv pip install -e . --torch-backend=auto
fi

.venv/bin/python - <<'PY'
import platform

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.cpu import _get_isa_hint
from vllm.platforms import CpuArchEnum, current_platform

machine = platform.machine().lower()
if not machine.startswith("riscv"):
    raise RuntimeError(f"expected RISC-V host, got {machine}")

arch = current_platform.get_cpu_architecture()
if arch != CpuArchEnum.RISCV:
    raise RuntimeError(f"vLLM did not detect RISC-V CPU architecture: {arch}")

if _get_isa_hint(torch.bfloat16) != "rvv":
    raise RuntimeError("CPUWNA16 Python ISA hint did not resolve to 'rvv'")

if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "cpu_gemm_wna16"):
    raise RuntimeError("torch.ops._C.cpu_gemm_wna16 is not registered")


def pack_w4_weight(weight: torch.Tensor, zero_points: torch.Tensor | None) -> torch.Tensor:
    # cpu_gemm_wna16 expects [N / 16, K * 2] for 4-bit packed int32 weights.
    k_size, n_size = weight.shape
    packed = torch.empty((n_size // 16, k_size * 2), dtype=torch.int32)
    for n_block in range(n_size // 16):
        for k_idx in range(k_size):
            for pack_idx in range(2):
                value = 0
                for nibble_idx in range(8):
                    n_idx = n_block * 16 + pack_idx * 8 + nibble_idx
                    if zero_points is None:
                        stored = int(weight[k_idx, n_idx].item()) + 8
                    else:
                        stored = int(weight[k_idx, n_idx].item()) + int(
                            zero_points[n_idx].item()
                        )
                    value |= (stored & 0xF) << (nibble_idx * 4)
                if value >= 2**31:
                    value -= 2**32
                packed[n_block, k_idx * 2 + pack_idx] = value
    return packed


def pack_w4_zeros(zero_points: torch.Tensor) -> torch.Tensor:
    packed = torch.empty((1, zero_points.numel() // 8), dtype=torch.int32)
    for pack_idx in range(zero_points.numel() // 8):
        value = 0
        for nibble_idx in range(8):
            value |= (int(zero_points[pack_idx * 8 + nibble_idx].item()) & 0xF) << (
                nibble_idx * 4
            )
        if value >= 2**31:
            value -= 2**32
        packed[0, pack_idx] = value
    return packed


def run_case(has_zp: bool) -> None:
    dtype = torch.bfloat16
    m_size, k_size, n_size = 3, 64, 32
    x = ((torch.arange(m_size * k_size).reshape(m_size, k_size) % 7) - 3).to(dtype)
    weight = ((torch.arange(k_size * n_size).reshape(k_size, n_size) % 15) - 7).to(
        torch.int32
    )
    scales = torch.ones((1, n_size), dtype=dtype)

    if has_zp:
        zero_points = torch.full((n_size,), 8, dtype=torch.int32)
        q_weight = pack_w4_weight(weight, zero_points)
        q_zeros = pack_w4_zeros(zero_points)
    else:
        q_weight = pack_w4_weight(weight, None)
        q_zeros = None

    out = ops.cpu_gemm_wna16(
        input=x,
        q_weight=q_weight,
        scales=scales,
        zeros=q_zeros,
        g_idx=None,
        bias=None,
        pack_factor=8,
        isa_hint="rvv",
    )
    ref = (x.float() @ weight.float()).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


run_case(has_zp=False)
run_case(has_zp=True)
print("cpu_gemm_wna16 RVV W4A16 smoke test passed")
PY
