"""Triton kernels for identity and block-routed SwiGLU activations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    x_fp32 = x.to(tl.float32)
    return x_fp32 * tl.sigmoid(x_fp32)


@triton.jit
def _baseline_kernel(gate_up, out, n_elements: tl.constexpr, width: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // width
    col = offsets - row * width
    gate = tl.load(gate_up + row * (2 * width) + col, mask=mask)
    up = tl.load(gate_up + row * (2 * width) + width + col, mask=mask)
    tl.store(out + offsets, _silu(gate) * up, mask=mask)


@triton.jit
def _routed_kernel(gate_up, routes, out, n_elements: tl.constexpr,
                   width: tl.constexpr, n_route_blocks: tl.constexpr,
                   ROUTE_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // width
    col = offsets - row * width
    route_block = col // ROUTE_BLOCK
    lane = col - route_block * ROUTE_BLOCK
    # The route is stored as a byte-sized cyclic shift. With a power-of-two
    # route block, selection is one load, one add and one bit mask.
    shift = tl.load(routes + row * n_route_blocks + route_block, mask=mask)
    partner_lane = (lane + shift) & (ROUTE_BLOCK - 1)
    partner_col = route_block * ROUTE_BLOCK + partner_lane

    gate = tl.load(gate_up + row * (2 * width) + partner_col, mask=mask)
    up = tl.load(gate_up + row * (2 * width) + width + col, mask=mask)
    tl.store(out + offsets, _silu(gate) * up, mask=mask)


def baseline_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU over a fused [gate, up] projection."""
    if not gate_up.is_cuda:
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    width = gate_up.shape[-1] // 2
    flat = gate_up.reshape(-1, 2 * width).contiguous()
    out = torch.empty((flat.shape[0], width), device=flat.device, dtype=flat.dtype)
    n_elements = out.numel()
    _baseline_kernel[(triton.cdiv(n_elements, 256),)](
        flat, out, n_elements, width, BLOCK=256
    )
    return out.reshape(*gate_up.shape[:-1], width)


def routed_swiglu(gate_up: torch.Tensor, routes: torch.Tensor,
                  route_block: int = 32) -> torch.Tensor:
    """Pair gate lanes with up lanes using a route per token and channel block."""
    width = gate_up.shape[-1] // 2
    if width % route_block:
        raise ValueError(f"width {width} must be divisible by route_block {route_block}")
    flat = gate_up.reshape(-1, 2 * width).contiguous()
    n_route_blocks = width // route_block
    routes = routes.reshape(flat.shape[0], n_route_blocks).contiguous()
    if not gate_up.is_cuda:
        return routed_reference(flat, routes, route_block).reshape(
            *gate_up.shape[:-1], width
        )
    out = torch.empty((flat.shape[0], width), device=flat.device, dtype=flat.dtype)
    n_elements = out.numel()
    _routed_kernel[(triton.cdiv(n_elements, 256),)](
        flat, routes, out, n_elements, width, n_route_blocks,
        ROUTE_BLOCK=route_block, BLOCK=256,
    )
    return out.reshape(*gate_up.shape[:-1], width)


def routed_reference(gate_up: torch.Tensor, routes: torch.Tensor,
                     route_block: int = 32) -> torch.Tensor:
    """Plain PyTorch reference used only for correctness tests."""
    width = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :width], gate_up[..., width:]
    rows = gate.shape[0]
    blocks = width // route_block
    lanes = torch.arange(route_block, device=gate.device)
    indices = (lanes.view(1, 1, -1) + routes.long().unsqueeze(-1))
    indices = (indices % route_block).reshape(rows, width)
    block_offsets = (
        torch.arange(blocks, device=gate.device) * route_block
    ).view(1, blocks, 1).expand(rows, -1, route_block).reshape(rows, width)
    paired_gate = torch.gather(gate, 1, indices + block_offsets)
    return torch.nn.functional.silu(paired_gate) * up
