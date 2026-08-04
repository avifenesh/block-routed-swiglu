# Block-routed SwiGLU experiment

This is a falsification-first prototype for adding input-dependent gate/up
pairings without changing the two dominant MLP matrix multiplications.

The performance gate compares three otherwise identical BF16 MLP blocks:

1. ordinary fused SwiGLU activation;
2. the routed kernel with identity routes (pure machinery overhead);
3. the routed kernel with mixed routes.

Default target shape is `d_model=4096`, `width=14336`, using 32-channel blocks
and four byte-encoded within-block cyclic shifts. The initial pass criterion is less
than 2% median whole-block latency overhead at useful token counts, no
correctness failures, and successful CUDA graph capture.

## H100 PCIe result (2026-07-22)

The GPU feasibility test passed after increasing routing granularity to 64
channels and interleaving measurements to remove ordering/clock bias:

- 512 tokens: `+0.73%` mean routed whole-block latency (`5 x 60` samples).
- 4096 tokens: `+0.83%` mean routed whole-block latency (`5 x 60` samples).
- CUDA graph capture passed and routed execution added no peak allocation.
- Identity and mixed routes matched their PyTorch references in BF16.

The equal-parameter synthetic capability test did **not** show a model-quality
gain across five seeds:

- structured held-out combinations: baseline `66.06%`, routed `63.38%`;
- random lookup held-out combinations: baseline `50.28%`, routed `50.42%`.

So the inexpensive routing mechanism is viable, but this routing/training rule
does not justify a language-model experiment yet. The next useful experiment is
to distill temporary training-only specialists into an unchanged dense MLP, or
to test a pairing objective that explicitly rewards reusable cross-channel
composition before paying for a 300M-parameter run.

Raw evidence is in `benchmark_interleaved_h100_b64.json` and
`structured_factor_results.json`. Earlier kernel variants and the first
confounded learning run are retained as negative controls
(`benchmark_h100*.json` for the route-block sweep and shift variants;
`structured_results.json` for the confounded learning run, in which both
models sat at chance on held-out pairs).

## Reproduce

Requires PyTorch with CUDA (results above are H100 PCIe, BF16; the
correctness test also runs on CPU in FP32).

```bash
# numerical correctness of the routed kernel vs references
python test_correctness.py

# performance gate: route-block granularity sweep
python benchmark.py --route-block 64 --output benchmark_b64.json

# interleaved confirmation at the chosen granularity (the headline numbers)
python benchmark_interleaved.py --route-block 64

# capability gate, 5 seeds (the headline 66.06% vs 63.38%)
python structured_test.py --output structured_factor_results.json
```

## License

MIT.
