"""Interleaved confirmation benchmark for the best route granularity."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from paired_swiglu import baseline_swiglu, routed_swiglu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_interleaved.json")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--route-block", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()

    torch.manual_seed(17)
    d_model, width = 4096, 14336
    dtype, device = torch.bfloat16, "cuda"
    w_gate_up = torch.randn(2 * width, d_model, device=device, dtype=dtype) / d_model**0.5
    w_down = torch.randn(d_model, width, device=device, dtype=dtype) / width**0.5
    output = {"gpu": torch.cuda.get_device_name(), "cases": []}

    for tokens in args.tokens:
        x = torch.randn(tokens, d_model, device=device, dtype=dtype)
        route_shape = (tokens, width // args.route_block)
        zero = torch.zeros(route_shape, device=device, dtype=torch.uint8)
        shifts = torch.tensor(
            (0, 1, args.route_block // 4, args.route_block // 2),
            device=device, dtype=torch.uint8,
        )
        mixed = shifts[torch.randint(0, 4, route_shape, device=device)]

        def baseline():
            return F.linear(baseline_swiglu(F.linear(x, w_gate_up)), w_down)

        def identity():
            return F.linear(routed_swiglu(
                F.linear(x, w_gate_up), zero, args.route_block
            ), w_down)

        def routed():
            return F.linear(routed_swiglu(
                F.linear(x, w_gate_up), mixed, args.route_block
            ), w_down)

        functions = {"baseline": baseline, "identity": identity, "routed": routed}
        for fn in functions.values():
            for _ in range(10):
                fn()
        torch.cuda.synchronize()

        trial_rows = []
        rng = random.Random(1000 + tokens)
        for trial in range(args.trials):
            samples = {name: [] for name in functions}
            order = list(functions) * args.samples
            rng.shuffle(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
            medians = {name: statistics.median(v) for name, v in samples.items()}
            medians["identity_overhead_pct"] = 100 * (
                medians["identity"] / medians["baseline"] - 1
            )
            medians["routed_overhead_pct"] = 100 * (
                medians["routed"] / medians["baseline"] - 1
            )
            trial_rows.append(medians)
        case = {
            "tokens": tokens,
            "route_block": args.route_block,
            "trials": trial_rows,
            "mean_identity_overhead_pct": statistics.mean(
                row["identity_overhead_pct"] for row in trial_rows
            ),
            "mean_routed_overhead_pct": statistics.mean(
                row["routed_overhead_pct"] for row in trial_rows
            ),
            "stdev_routed_overhead_pct": statistics.stdev(
                row["routed_overhead_pct"] for row in trial_rows
            ),
        }
        output["cases"].append(case)
        print(json.dumps(case), flush=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
