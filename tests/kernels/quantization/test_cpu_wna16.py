# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU W4A16 GEMM kernel tests."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.cpu import _get_isa_hint
from vllm.platforms import CpuArchEnum, current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if current_platform.get_cpu_architecture() != CpuArchEnum.RISCV:
    pytest.skip("skipping RISC-V-only CPU WNA16 tests", allow_module_level=True)

if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "cpu_gemm_wna16"):
    pytest.skip("cpu_gemm_wna16 op not available", allow_module_level=True)


def _to_signed_int32(value: int) -> int:
    if value >= 2**31:
        return value - 2**32
    return value


def _pack_w4_weight(
    weight: torch.Tensor,
    zero_points: torch.Tensor | None,
) -> torch.Tensor:
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
                packed[n_block, k_idx * 2 + pack_idx] = _to_signed_int32(value)

    return packed


def _pack_w4_zeros(zero_points: torch.Tensor) -> torch.Tensor:
    packed = torch.empty((1, zero_points.numel() // 8), dtype=torch.int32)

    for pack_idx in range(zero_points.numel() // 8):
        value = 0
        for nibble_idx in range(8):
            value |= (int(zero_points[pack_idx * 8 + nibble_idx].item()) & 0xF) << (
                nibble_idx * 4
            )
        packed[0, pack_idx] = _to_signed_int32(value)

    return packed


def test_cpu_gemm_wna16_rvv_isa_hint():
    assert _get_isa_hint(torch.bfloat16) == "rvv"


@pytest.mark.parametrize("has_zp", [False, True])
def test_cpu_gemm_wna16_rvv_w4a16(has_zp: bool):
    dtype = torch.bfloat16
    m_size, k_size, n_size = 3, 64, 32

    x = ((torch.arange(m_size * k_size).reshape(m_size, k_size) % 7) - 3).to(
        dtype
    )
    weight = ((torch.arange(k_size * n_size).reshape(k_size, n_size) % 15) - 7).to(
        torch.int32
    )
    scales = torch.ones((1, n_size), dtype=dtype)

    if has_zp:
        zero_points = torch.full((n_size,), 8, dtype=torch.int32)
        q_weight = _pack_w4_weight(weight, zero_points)
        q_zeros = _pack_w4_zeros(zero_points)
    else:
        q_weight = _pack_w4_weight(weight, None)
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
