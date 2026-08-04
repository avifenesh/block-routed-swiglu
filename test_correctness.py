from __future__ import annotations

import torch

from paired_swiglu import baseline_swiglu, routed_reference, routed_swiglu


def run(device: str) -> None:
    torch.manual_seed(7)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for rows, width, block in ((1, 32, 32), (7, 64, 32), (19, 128, 32)):
        x = torch.randn(rows, 2 * width, device=device, dtype=dtype)
        routes0 = torch.zeros(rows, width // block, device=device, dtype=torch.uint8)
        choices = torch.tensor((0, 1, 8, 16), device=device, dtype=torch.uint8)
        routes = choices[torch.randint(0, 4, routes0.shape, device=device)]

        expected = torch.nn.functional.silu(x[:, :width]) * x[:, width:]
        baseline = baseline_swiglu(x)
        identity = routed_swiglu(x, routes0, block)
        mixed = routed_swiglu(x, routes, block)
        mixed_ref = routed_reference(x, routes, block)

        atol = 5e-2 if dtype == torch.bfloat16 else 1e-6
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(baseline, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(identity, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(mixed, mixed_ref, atol=atol, rtol=rtol)
    print(f"correctness: PASS ({device})")


if __name__ == "__main__":
    run("cuda" if torch.cuda.is_available() else "cpu")
