# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear.mixed_precision import (
    CPUWNA16LinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision import cpu as cpu_kernel
from vllm.platforms import CpuArchEnum
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.cpu_test


def _config(weight_type, group_size: int = 32) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(64, 64),
        partition_weight_shape=(64, 64),
        weight_type=weight_type,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=weight_type in (scalar_types.uint4, scalar_types.uint8),
        has_g_idx=False,
    )


@pytest.mark.parametrize(
    "weight_type",
    [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8,
        scalar_types.uint8b128,
    ],
)
def test_cpu_wna16_accepts_w4_and_w8(weight_type):
    with patch(
        "vllm.model_executor.kernels.linear.mixed_precision.cpu.current_platform"
    ) as platform:
        platform.is_cpu.return_value = True

        can_implement, reason = CPUWNA16LinearKernel.can_implement(
            _config(weight_type)
        )

    assert can_implement, reason


def test_cpu_wna16_rejects_odd_group_size():
    with patch(
        "vllm.model_executor.kernels.linear.mixed_precision.cpu.current_platform"
    ) as platform:
        platform.is_cpu.return_value = True

        can_implement, reason = CPUWNA16LinearKernel.can_implement(
            _config(scalar_types.uint8b128, group_size=3)
        )

    assert not can_implement
    assert "multiples of 2" in reason


def test_cpu_wna16_riscv_accepts_only_w8():
    with patch(
        "vllm.model_executor.kernels.linear.mixed_precision.cpu.current_platform"
    ) as platform:
        platform.is_cpu.return_value = True
        platform.get_cpu_architecture.return_value = CpuArchEnum.RISCV

        can_w4, w4_reason = CPUWNA16LinearKernel.can_implement(
            _config(scalar_types.uint4b8)
        )
        can_w8, w8_reason = CPUWNA16LinearKernel.can_implement(
            _config(scalar_types.uint8b128)
        )

    assert not can_w4
    assert "8-bit" in w4_reason
    assert can_w8, w8_reason


def test_cpu_wna16_w8_passes_pack_factor_4():
    kernel = object.__new__(CPUWNA16LinearKernel)
    kernel.config = _config(scalar_types.uint8b128)
    kernel.w_q_name = "weight_packed"
    kernel.w_s_name = "weight_scale"
    kernel.w_zp_name = None
    kernel.w_gidx_name = None

    layer = torch.nn.Module()
    layer.weight_packed = torch.empty((4, 128), dtype=torch.int32)
    layer.weight_scale = torch.empty((1, 64), dtype=torch.bfloat16)
    layer.isa_hint = "rvv"
    x = torch.empty((1, 64), dtype=torch.bfloat16)

    with patch.object(cpu_kernel.ops, "cpu_gemm_wna16") as gemm:
        gemm.return_value = torch.empty((1, 64), dtype=torch.bfloat16)
        kernel.apply_weights(layer, x)

    assert gemm.call_args.kwargs["pack_factor"] == 4
    assert gemm.call_args.kwargs["isa_hint"] == "rvv"
