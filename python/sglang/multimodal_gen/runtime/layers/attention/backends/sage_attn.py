# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


import torch
from sageattention import sageattn

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        # SageAttention's CUDA kernels assume dense NHD/HND strides even though
        # the public implementation only checks that the last dimension is
        # contiguous. Fused MiniMax-H3 QKV projection and Ulysses both return
        # split views whose last dimension is contiguous but whose sequence and
        # head strides still include all three QKV tensors.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = sageattn(
            query,
            key,
            value,
            # since input is (batch_size, seq_len, head_num, head_dim)
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            return_lse=return_softmax_lse,
        )
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run packed attention without allowing tokens to cross segments.

        SageAttention's architecture-dispatched ``sageattn`` entry point uses
        the optimized SM-specific implementation, while its native varlen API
        currently uses a separate Triton FP16-PV path. Execute each packed
        segment as a batch-of-one so Hopper inputs continue to use the SM90
        implementation and retain the exact packed-sequence boundaries.
        """
        del max_seqlen
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != query.shape[0]:
            raise ValueError(
                "cu_seqlens must start at 0 and end at the packed query length"
            )
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            raise ValueError("packed SageAttention requires matching Q/K/V lengths")
        if any(stop < start for start, stop in zip(bounds[:-1], bounds[1:])):
            raise ValueError("cu_seqlens must be monotonically non-decreasing")

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            segment = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )
            output[start:stop].copy_(segment[0])
        return output
