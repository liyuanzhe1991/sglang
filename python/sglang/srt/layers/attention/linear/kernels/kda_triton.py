from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu

_is_hip = is_hip()
if _is_hip:
    from sglang.kernels.ops.attention.linear import kda_gfx950_prefill

if not is_cpu():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )


# FlashInfer's SM120 KDA prefill takes one K = V = 128 sequence with the bounded
# gate; shorter extends gain nothing over the Triton chain.
_FI_SM120_KDA_MIN_TOKENS = 1024
_FI_SM120_KDA_DIM = 128
_FI_SM120_KDA_CHUNK = 64
_fi_sm120_kda_prefill = None
_fi_sm120_kda_params = {}
_fi_sm120_kda_track = (None, None)
_fi_sm120_kda_bufs = None


def _fi_sm120_kda_buffers(q, beta, ssm_states):
    """Operands the kernel is always called on.

    FlashInfer caches its call plans and tensor views on buffer addresses and
    keeps those buffers alive, so per-call tensors would pile up in its caches.
    """
    global _fi_sm120_kda_bufs
    bufs = _fi_sm120_kda_bufs
    if (
        bufs is None
        or bufs[0].shape[1] < q.shape[1]
        or bufs[0].shape[2:] != q.shape[2:]
        or bufs[5].dtype != ssm_states.dtype
    ):

        def new(shape, like):
            return torch.empty(shape, dtype=like.dtype, device=like.device)

        bufs = _fi_sm120_kda_bufs = (
            *(new(q.shape, q) for _ in range(4)),
            new(q.shape[:3], beta),
            new((1, *ssm_states.shape[1:]), ssm_states),
            new(q.shape, q),
        )
    return bufs


def _fi_sm120_kda_snapshot_tokens(track_chunk_idx: torch.Tensor) -> int:
    """Tokens of the single sequence before its tracked chunk boundary.

    One host read per forward: every layer of a forward passes the same tensor.
    """
    global _fi_sm120_kda_track
    if _fi_sm120_kda_track[0] is not track_chunk_idx:
        _fi_sm120_kda_track = (track_chunk_idx, track_chunk_idx.tolist())
    return _fi_sm120_kda_track[1][0] * _FI_SM120_KDA_CHUNK


def _load_fi_sm120_kda_prefill():
    """FlashInfer's SM120 KDA prefill entry, or False where it cannot run."""
    global _fi_sm120_kda_prefill
    if _fi_sm120_kda_prefill is None:
        run = False
        if (
            envs.SGLANG_KDA_PREFILL_FLASHINFER_SM120.get()
            and is_cuda()
            and torch.cuda.get_device_capability() == (12, 0)
        ):
            from flashinfer import kda_kernels

            run = kda_kernels.run_kda_prefill_sm120 or False
        _fi_sm120_kda_prefill = run
    return _fi_sm120_kda_prefill


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    # XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode; route XPU to the
    # non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update),
    # the same fallback CPU/NPU use. Batched decode is handled via query_start_loc.
    supports_packed_decode: bool = not is_cpu() and not is_npu() and not is_xpu()
    supports_packed_decode_safe_gate: bool = is_cuda()
    supports_fused_chain_verify: bool = not is_cpu() and not is_npu()
    supports_track_state_snapshot: bool = True

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # KDA ReplaySSM buffered decode: drop-in for the packed decode, same
        # args plus the three per-layer ring caches + the per-row write cursor
        # (and optional radix-track force-flush). Uses the gate-generic kernel
        # with is_kda=True (per-K gate); g_cache is [num_slots, HV, L, K].
        # When any ring tensor / cursor is None (flag off) we fall through to
        # the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            lower_bound is None
            and replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            if lower_bound is not None:
                raise NotImplementedError(
                    "KDA safe gate (lower_bound) is not implemented in the "
                    "ReplaySSM decode kernel; disable --enable-linear-replayssm."
                )
            K = ssm_states.shape[-1]  # ssm_states: [num_slots, HV, V, K]
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(B, num_v_heads, K).contiguous(),
                b=b.reshape(B, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, K).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
            )
            return out.transpose(0, 1)

        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower_bound,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=lower_bound,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        # fused ReplaySSM ring-write (dense verify only; off elsewhere).
        cache_ring: bool = False,
        replayssm_rawv: Optional[torch.Tensor] = None,
        replayssm_rawk: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        replayssm_beta: Optional[torch.Tensor] = None,
        ring_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # KDA MTP / speculative-decode verify via the fused KDA kernel (IS_KDA=True),
        # mirroring the GDN triton verify path. Reads the committed state, writes
        # per-draft-token intermediate states to the scratch buffer, does NOT mutate
        # the committed pool (disable_state_update=True), and handles chain + tree
        # (retrieve_parent_token). The verify kernel for the Triton / CuTe DSL KDA
        # decode backends, and the reference the KDA correctness tests assert against.
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
            cache_ring=cache_ring,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
            ring_indices=ring_indices,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fi_run = _load_fi_sm120_kda_prefill()
        tokens = q.shape[1]
        track_state = kwargs.get("track_state")
        track_h_src = kwargs.get("track_ssm_h_src")
        # Tokens before the chunk-boundary state the prefix cache asks for.
        snapshot_tokens = 0
        if (
            fi_run
            and query_start_loc.numel() == 2
            and tokens >= _FI_SM120_KDA_MIN_TOKENS
            and q.shape[-1] == v.shape[-1] == _FI_SM120_KDA_DIM
            and q.shape[2] == v.shape[2]
            and q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype == torch.bfloat16
            and beta_is_raw
            and A_log is not None
            and dt_bias is not None
            and lower_bound is not None
            and -5.0 <= lower_bound < 0.0
            and not kwargs.get("is_spec_decode")
        ):
            if track_state is not None:
                snapshot_tokens = _fi_sm120_kda_snapshot_tokens(
                    kwargs["track_chunk_idx"]
                )
            elif track_h_src is not None and track_h_src.numel() > 0:
                fi_run = False
            # Both parts of a split extend must be ordinary multi-token calls.
            if track_state is not None and not (
                _FI_SM120_KDA_CHUNK <= snapshot_tokens <= tokens - _FI_SM120_KDA_CHUNK
            ):
                fi_run = False
        else:
            fi_run = False
        if fi_run:
            bq, bk, bv, bg, bbeta, state, bout = _fi_sm120_kda_buffers(
                q, beta, ssm_states
            )
            operands = []
            for buf, src in ((bq, q), (bk, k), (bv, v), (bg, g), (bbeta, beta)):
                operands.append(buf[:, :tokens].copy_(src[:, :tokens]))
            key = (A_log.data_ptr(), dt_bias.data_ptr())
            params = _fi_sm120_kda_params.get(key)
            if params is None:
                params = _fi_sm120_kda_params[key] = (
                    A_log.reshape(-1).float().contiguous(),
                    dt_bias.reshape(-1, _FI_SM120_KDA_DIM).float().contiguous(),
                )
            slot = cache_indices[:1].to(torch.int64)
            # The fused variant keeps the state in the pool's dtype.
            torch.index_select(ssm_states, 0, slot, out=state)
            for start, end in ((0, snapshot_tokens), (snapshot_tokens, tokens)):
                if start == end:
                    continue
                fi_run(
                    q=operands[0][:, start:end],
                    k=operands[1][:, start:end],
                    v=operands[2][:, start:end],
                    g=operands[3][:, start:end],
                    beta=operands[4][:, start:end],
                    A_log=params[0],
                    dt_bias=params[1],
                    lower_bound=float(lower_bound),
                    initial_state=state,
                    final_state=state,
                    output=bout[:, start:end],
                    variant="fused",
                )
                if end == snapshot_tokens:
                    track_state[:1] = state
            ssm_states.index_copy_(0, slot, state)
            out = bout[:, :tokens].clone()
            return (out, None) if return_intermediate_states else out
        if _is_hip:
            out = kda_gfx950_prefill.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )
            if out is not None:
                return out
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            output_intermediate_states=return_intermediate_states,
            track_state=kwargs.get("track_state"),
            track_chunk_idx=kwargs.get("track_chunk_idx"),
            chunk_indices=kwargs.get("chunk_indices"),
            chunk_offsets=kwargs.get("chunk_offsets"),
        )
