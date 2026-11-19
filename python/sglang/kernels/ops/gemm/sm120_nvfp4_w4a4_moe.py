from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import tempfile
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# sha256 of the flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_nvfp4_w4a4.py
# files whose argument conventions and GEMM1 epilogue this module was validated
# against.
_VALIDATED_KERNEL_SHA256 = {
    # flashinfer 0.7.4
    "b91b043293176c7cb1e8833c1a86a0197f2d81416071fcbbb34b6b019812127f",
}

# Token counts are padded up to these row counts so that a process compiles a
# handful of kernels; padding slots carry expert id -1, which the router drops.
_SMALL_BUCKETS = (64, 256)
_BUCKET_STEP = 512
# Buckets above this compile on their first use: each keeps its own workspace.
_WARM_UP_MAX_TOKENS = 4096

# GEMM1 epilogue of FlashInfer's kernel: dequantized gate / up before SiLU.
_PLAIN_SWIGLU = (
    "                        gv = acc[i] * scale\n"
    "                        uv = acc[i + 2 * VM] * scale\n"
)
# The same with CUTLASS's SwigluBias clamp (alpha 1, beta 0):
# gate = min(gate, limit), up = clamp(up, -limit, limit).
_CLAMPED_SWIGLU = (
    "                        gv = -cute.arch.fmax(-(acc[i] * scale), F32(-{limit}))\n"
    "                        uv = cute.arch.fmax(\n"
    "                            -cute.arch.fmax(\n"
    "                                -(acc[i + 2 * VM] * scale), F32(-{limit})\n"
    "                            ),\n"
    "                            F32(-{limit}),\n"
    "                        )\n"
)

_buffers: dict = {}
_warmed_up: set = set()


def _bucket(num_tokens: int) -> int:
    for rows in _SMALL_BUCKETS:
        if num_tokens <= rows:
            return rows
    return -(-num_tokens // _BUCKET_STEP) * _BUCKET_STEP


@functools.lru_cache(maxsize=None)
def _kernel(swiglu_limit: Optional[float]):
    """FlashInfer's SM12x W4A4 MoE module, or a copy with the SwiGLU clamp."""
    import flashinfer
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_nvfp4_w4a4

    with open(moe_nvfp4_w4a4.__file__, "rb") as f:
        source = f.read()
    if hashlib.sha256(source).hexdigest() not in _VALIDATED_KERNEL_SHA256:
        raise RuntimeError(
            "SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4 is not validated against the "
            f"SM12x W4A4 MoE kernel of flashinfer {flashinfer.__version__}"
        )
    if swiglu_limit is None:
        return moe_nvfp4_w4a4
    # FlashInfer's kernel has no clamp argument and CuTe DSL compiles from the
    # source file, so the clamped kernel is a patched copy of that file.
    source = source.decode()
    if source.count(_PLAIN_SWIGLU) != 1:
        raise RuntimeError(
            "FlashInfer's SM12x W4A4 MoE epilogue changed; cannot add the SwiGLU clamp"
        )
    source = source.replace(
        _PLAIN_SWIGLU, _CLAMPED_SWIGLU.format(limit=repr(float(swiglu_limit)))
    )
    # Named by content: stage processes and restarts share one file.
    name = "sglang_moe_nvfp4_w4a4_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), name + ".py")
    if not os.path.exists(path):
        with tempfile.NamedTemporaryFile(
            "w", dir=os.path.dirname(path), suffix=".py", delete=False
        ) as f:
            f.write(source)
        os.replace(f.name, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def w4a4_swiglu_limit(swiglu_params) -> Optional[float]:
    """CUTLASS per-expert SwiGLU (alpha, beta, limit) -> the kernel's clamp limit."""
    alpha, beta, limit = swiglu_params
    if limit is None:
        return None
    if not (
        bool((alpha == 1.0).all())
        and bool((beta == 0.0).all())
        and bool((limit == limit[0]).all())
    ):
        raise RuntimeError(
            "The SM12x W4A4 MoE kernel supports one SwiGLU limit with "
            "alpha 1 and beta 0"
        )
    return float(limit[0])


def nvfp4_w4a4_moe(x, w13, w13_sf, w2, w2_sf, params, ids, routing):
    """Gated NVFP4 x NVFP4 routed experts on SM120; w13 rows are [gate; up].

    params = (w13_gs, a13_gs, w2_gs, a2_gs, swiglu_limit): per-expert weight
    scales, one-element reciprocal activation scales, optional clamp limit.
    """
    w13_gs, a13_gs, w2_gs, a2_gs, swiglu_limit = params
    kernel = _kernel(swiglu_limit)
    num_tokens, hidden = x.shape
    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=torch.bfloat16, device=x.device)
    num_experts, inter = w2.shape[0], w13.shape[1] // 2
    top_k = ids.shape[1]
    rows = _bucket(num_tokens)
    key = (x.device.index, rows, hidden, inter, num_experts, top_k, swiglu_limit)
    buf = _buffers.get(key)
    if buf is None:
        logger.info("SM12x W4A4 MoE: compiling the kernels for %d rows", rows)
        # The padded operands are updated in place by later forwards, which
        # may run outside the inference mode of the first one.
        with torch.inference_mode(False):
            buf = _buffers[key] = {
                "workspace": kernel.allocate_workspace(
                    rows, hidden, inter, num_experts, top_k, True, x.device
                ),
                "x": torch.zeros((rows, hidden), dtype=torch.bfloat16, device=x.device),
                "ids": torch.full(
                    (rows, top_k), -1, dtype=torch.int32, device=x.device
                ),
                "routing": torch.zeros(
                    (rows, top_k), dtype=torch.float32, device=x.device
                ),
                "used": 0,
            }

    if rows == num_tokens:
        x_in = x.contiguous()
        ids_in = ids.to(torch.int32).contiguous()
        routing_in = routing.contiguous()
    else:
        x_in, ids_in, routing_in = buf["x"], buf["ids"], buf["routing"]
        x_in[:num_tokens].copy_(x)
        ids_in[:num_tokens].copy_(ids)
        routing_in[:num_tokens].copy_(routing)
        if buf["used"] > num_tokens:
            ids_in[num_tokens : buf["used"]].fill_(-1)
        buf["used"] = num_tokens

    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device)
    kernel.run_moe_w4a4(
        x_in,
        ids_in,
        routing_in,
        w13,
        w13_sf.view(torch.float8_e4m3fn).view(num_experts, 2 * inter, hidden // 16),
        w13_gs,
        w2,
        w2_sf.view(torch.float8_e4m3fn).view(num_experts, hidden, inter // 16),
        w2_gs,
        a13_gs,
        a2_gs,
        out,
        buf["workspace"],
    )
    return out[:num_tokens]


def warm_up_w4a4_moe(w13, w13_sf, w2, w2_sf, params, top_k, max_tokens):
    """Load the kernel and compile its token buckets up to max_tokens.

    Done at load: CUTLASS cannot take over a forward once w13 is [gate; up],
    and a bucket compiled inside a forward stalls it for about a second.
    """
    _kernel(params[4])
    if not (w13.is_contiguous() and w2.is_contiguous()):
        raise RuntimeError("The SM12x W4A4 MoE kernel needs contiguous weights")
    hidden = w2.shape[1]
    max_tokens = min(max_tokens, _WARM_UP_MAX_TOKENS)
    key = (w13.device.index, hidden, w13.shape[1], w2.shape[0], top_k, params[4])
    if (key, max_tokens) in _warmed_up:
        return
    rows = 0
    while rows < max_tokens:
        rows = _bucket(rows + 1)
        x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=w13.device)
        ids = torch.full((rows, top_k), -1, dtype=torch.int32, device=w13.device)
        nvfp4_w4a4_moe(x, w13, w13_sf, w2, w2_sf, params, ids, ids.float())
    _warmed_up.add((key, max_tokens))
