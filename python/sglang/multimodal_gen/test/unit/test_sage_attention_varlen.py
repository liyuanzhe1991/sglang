# SPDX-License-Identifier: Apache-2.0
"""Packed-sequence contracts for the SageAttention diffusion backend."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch


def _load_sage_backend(sageattn: Mock):
    stub = types.ModuleType("sageattention")
    stub.sageattn = sageattn
    module_path = (
        Path(__file__).parents[2] / "runtime/layers/attention/backends/sage_attn.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_sage_attention_varlen_backend", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"sageattention": stub}):
        spec.loader.exec_module(module)
    return module


def _impl(sageattn: Mock):
    module = _load_sage_backend(sageattn)
    return module.SageAttentionImpl(
        num_heads=2,
        head_size=4,
        causal=False,
        softmax_scale=0.5,
    )


def test_forward_varlen_runs_each_nonempty_segment_independently():
    calls = []

    def fake_sageattn(query, key, value, **kwargs):
        calls.append((query.clone(), key.clone(), value.clone(), kwargs))
        return query + 10 * key + 100 * value

    sageattn = Mock(side_effect=fake_sageattn)
    impl = _impl(sageattn)
    query = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
    key = query + 1000
    value = query + 2000

    output = impl.forward_varlen(
        query,
        key,
        value,
        cu_seqlens=torch.tensor([0, 3, 4, 6], dtype=torch.int32),
        cu_seqlens_host=(0, 3, 4, 6),
        max_seqlen=3,
    )

    torch.testing.assert_close(output, query + 10 * key + 100 * value)
    assert [call[0].shape[1] for call in calls] == [3, 1, 2]
    assert all(call[0].shape[0] == 1 for call in calls)
    assert all(call[3]["tensor_layout"] == "NHD" for call in calls)
    assert all(call[3]["sm_scale"] == 0.5 for call in calls)


def test_forward_varlen_skips_empty_trailing_segment():
    sageattn = Mock(side_effect=lambda query, key, value, **_kwargs: query)
    impl = _impl(sageattn)
    query = torch.randn(5, 2, 4)

    output = impl.forward_varlen(
        query,
        query,
        query,
        cu_seqlens=torch.tensor([0, 5, 5], dtype=torch.int32),
        cu_seqlens_host=(0, 5, 5),
        max_seqlen=5,
    )

    torch.testing.assert_close(output, query)
    sageattn.assert_called_once()


@pytest.mark.parametrize(
    "bounds",
    [
        (1, 5),
        (0, 4),
        (0, 4, 3, 5),
    ],
)
def test_forward_varlen_rejects_invalid_bounds(bounds):
    impl = _impl(Mock())
    query = torch.randn(5, 2, 4)

    with pytest.raises(ValueError, match="cu_seqlens"):
        impl.forward_varlen(
            query,
            query,
            query,
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
            cu_seqlens_host=bounds,
            max_seqlen=5,
        )
