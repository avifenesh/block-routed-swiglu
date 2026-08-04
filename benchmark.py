from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from paired_swiglu import baseline_swiglu, routed_swiglu


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def timed(fn, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
    }


def graph_capture_ok(fn) -> bool:
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        print(f"graph capture failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark.json")
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--width", type=int, default=14336)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 512, 4096])
    parser.add_argument("--route-block", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    dtype = torch.bfloat16
    d_model, width = args.d_model, args.width

    w_gate_up = torch.randn(2 * width, d_model, device=device, dtype=dtype) / d_model**0.5
    w_down = torch.randn(d_model, width, device=device, dtype=dtype) / width**0.5
    results: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "d_model": d_model,
        "width": width,
        "cases": [],
    }

    for tokens in args.tokens:
        x = torch.randn(tokens, d_model, device=device, dtype=dtype)
        route_shape = (tokens, width // args.route_block)
        identity_routes = torch.zeros(route_shape, device=device, dtype=torch.uint8)
        shifts = torch.tensor(
            (0, 1, args.route_block // 4, args.route_block // 2),
            device=device, dtype=torch.uint8,
        )
        mixed_routes = shifts[torch.randint(0, 4, route_shape, device=device)]

        def baseline_block():
            return F.linear(baseline_swiglu(F.linear(x, w_gate_up)), w_down)

        def routed_identity_block():
            return F.linear(routed_swiglu(
                F.linear(x, w_gate_up), identity_routes, args.route_block
            ), w_down)

        def routed_mixed_block():
            return F.linear(routed_swiglu(
                F.linear(x, w_gate_up), mixed_routes, args.route_block
            ), w_down)

        # Compile kernels and settle matmul autotuning before measurements.
        baseline_block(); routed_identity_block(); routed_mixed_block()
        torch.cuda.synchronize()
        repeats = 80 if tokens <= 32 else (40 if tokens <= 512 else 20)
        warmup = 10
        case = {"tokens": tokens}
        for name, fn in (
            ("baseline", baseline_block),
            ("routed_identity", routed_identity_block),
            ("routed_mixed", routed_mixed_block),
        ):
            torch.cuda.reset_peak_memory_stats()
            stats = timed(fn, warmup, repeats)
            stats["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            case[name] = stats
        base = case["baseline"]["p50_ms"]
        case["identity_overhead_pct"] = 100 * (case["routed_identity"]["p50_ms"] / base - 1)
        case["mixed_overhead_pct"] = 100 * (case["routed_mixed"]["p50_ms"] / base - 1)
        case["cuda_graph_capture"] = {
            "baseline": graph_capture_ok(baseline_block),
            "routed": graph_capture_ok(routed_mixed_block),
        }
        results["cases"].append(case)
        print(json.dumps(case))

    results["timestamp"] = time.time()
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
