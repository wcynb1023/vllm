# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform


def _to_signed_int32(value: int) -> int:
    if value >= 2**31:
        return value - 2**32
    return value


def _pack_w4_weight(weight: torch.Tensor) -> torch.Tensor:
    k_size, n_size = weight.shape
    assert n_size % 16 == 0

    packed = torch.empty((n_size // 16, k_size * 2), dtype=torch.int32)
    for n_block in range(n_size // 16):
        for k_idx in range(k_size):
            for pack_idx in range(2):
                packed_value = 0
                for nibble_idx in range(8):
                    n_idx = n_block * 16 + pack_idx * 8 + nibble_idx
                    stored = int(weight[k_idx, n_idx].item()) + 8
                    packed_value |= (stored & 0xF) << (nibble_idx * 4)
                packed[n_block, k_idx * 2 + pack_idx] = _to_signed_int32(
                    packed_value)
    return packed


@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.RISCV,
    reason="RISC-V only",
)
def test_cpu_gemm_wna16_vec_riscv_gptq_w4a16() -> None:
    from vllm import _custom_ops as ops

    assert hasattr(torch.ops._C, "cpu_gemm_wna16")

    dtype = torch.float16
    m_size = 3
    k_size = 64
    n_size = 32

    x = ((torch.arange(m_size * k_size) % 7) - 3).reshape(
        m_size, k_size).to(dtype)
    weight = ((torch.arange(k_size * n_size) % 16) - 8).reshape(
        k_size, n_size).to(torch.float32)

    q_weight = _pack_w4_weight(weight)
    scales = torch.ones((1, n_size), dtype=dtype)

    out = ops.cpu_gemm_wna16(
        input=x,
        q_weight=q_weight,
        scales=scales,
        zeros=None,
        g_idx=None,
        bias=None,
        pack_factor=8,
        isa_hint="vec",
    )

    ref = (x.float() @ weight).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-2)
