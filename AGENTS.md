# Development contract

- Preserve the labels-only dense Euclidean contract in README.md. Do not claim general sklearn API compatibility.
- Reference: scikit-learn 1.9.1, NumPy 2.3.5, `algorithm="auto"`, self-inclusive `min_samples`, EOM, alpha 1, no single root cluster. Do not silently switch to the contrib package or sklearn's brute path.
- Compare exact noise masks and partitions modulo cluster labels. Keep the explicit float32 precision counterexample; do not hide new mismatches behind tolerances, jitter or blanket xfails.
- Never introduce a CPU clustering fallback into a GPU backend. Scalar validation synchronizations are documented; clustering intermediates must stay on device.
- Run `uv run pytest -q`. Missing GPU prerequisites may skip tests; kernel failures must fail. Report CUDA skips separately from Metal passes.
- Benchmark with actual device completion, warmups and both resident and transfer-inclusive timings. Read benchmarks/RESULTS.md before making performance claims.
- Retain third-party attribution and ship THIRD_PARTY_NOTICES.md with wheels and source distributions.
