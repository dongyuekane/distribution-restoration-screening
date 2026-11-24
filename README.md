# Service-Block Exclusion for AC-Verified Restoration with Reverse Power Flow

Research code and numerical results for the fixed-slack radial feeder benchmark. The method propagates branch-flow bounds over blocks of service levels under fixed distributed generation. A block is excluded when its voltage or current bounds fail a necessary operating condition. A batched backward–forward AC sweep evaluates the remaining states.

## Run the confirmation experiment

Use Python 3.12 on macOS or Linux. NumPy is the only numerical dependency.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_cases.py
python experiment.py --split confirmation
python verify.py
```

The development split is available via `python experiment.py --split development`. It writes a fresh selection record to the output directory and preserves the archived selection. Repeated runs write under `generated/` by default. Use `--output-dir` for a different location. For offline use, place the exact MATPOWER case files in a local directory and specify `--case-dir`.

## Benchmark

The archived confirmation workload contains 18 radial topologies across two base feeders, 162 synthetic operating profiles, and 13,122 candidate service states. Fixed-width blocks send 2,486 states to the AC sweep and return the same highest AC-accepted service level as complete batching in all 162 profiles. Per-state classifications, numerical residuals, and raw timing measurements are in `archive/`. The comparison includes complete and descending AC batches plus one- and three-update pointwise bounds. Timing depends on hardware and the numerical runtime. The [archived runtime plot](figures/archived-runtime.png) summarizes the per-profile median timings.

The tests use balanced constant-PQ radial feeders with fixed source voltage and study-defined current limits. The floating-point experiment checks observed agreement with a common AC acceptance rule. The study's coupled transmission–distribution experiment uses a separate implementation and protocol.

## Inputs and provenance

`protocol.json` and `frozen_selection.json` record the benchmark protocol and archived method selection. The standard feeders are MATPOWER 8.1 `case33bw.m` and `case69.m`. The study adds five assumed tie lines to the 69-bus feeder. `fetch_cases.py` obtains the case files from the [upstream MATPOWER distribution](https://github.com/MATPOWER/matpower/tree/8.1/data) and verifies their archival SHA-256 hashes. MATPOWER documents its software and case-file terms in its [license notice](https://github.com/MATPOWER/matpower/blob/8.1/LICENSE). Topologies, operating scenarios and results under `archive/` come from the study benchmark.
