# OpenAI Sparse-Circuit Ablation and Progressive PLOT Reproduction

This release reproduces the July 15, 2026 experiment sequence:

1. mean-ablate the certified quote and bracket handles;
2. rerun PLOT over every remaining localized site to look for redundancy;
3. use the certified bracket binary readout to search progressively upstream;
4. test and localize the graded active-depth variable `D`;
5. validate the refined bracket model `X -> D -> R -> Y`.

The release contains a fast offline audit over frozen records and the code/configurations for complete GPU reruns. It does not contain model checkpoints.

## Main result

```text
X -> D -> R -> Y

D: active bracket depth in {1,2,3,4}
R: 0 when D=1; 1 when D>=2
Y: one-close versus two-close output
```

The validated neural pathway is:

```text
2.attn.resid_delta:1249 + 3.attn.resid_delta:1249
  -> 4.attn.act_in:1249
  -> 4.attn.resid_delta:1079
  -> output
```

See `REPORT.md` for the complete methods, failed attempts, metrics, caveats, and interpretation.

## Model acquisition

The two public OpenAI checkpoints are downloaded from:

```text
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo1/
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo2/
```

The implementation dependency is pinned to:

```text
https://github.com/openai/circuit_sparsity
commit dbf1fe0d27b76c19e10d2a715f28c2e5da535e08
```

`MODEL_ARTIFACTS.json` records every URL, byte count, and SHA256 checksum. The setup scripts verify downloads before use.

## Fast offline audit

Windows PowerShell:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m sparse_circuit_repro audit
.\.venv\Scripts\python.exe -m unittest discover tests
```

Linux/macOS:

```bash
bash scripts/setup.sh
.venv/bin/python -m sparse_circuit_repro audit
.venv/bin/python -m unittest discover tests
```

The setup command downloads both checkpoints and runs a clean-model smoke test. To audit the frozen records without downloading checkpoints, install the package and run `audit` directly.

## Full reruns

Run the complete sequence in order:

```powershell
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment all --cuda
```

Or run individual stages:

```powershell
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment necessity-quote --cuda
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment necessity-bracket --cuda
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment rediscover-quote --cuda
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment rediscover-bracket --cuda
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment progressive-rmid --cuda
.\.venv\Scripts\python.exe -m sparse_circuit_repro run --experiment graded-depth --cuda
```

Runs checkpoint their expensive scans and can be resumed by issuing the same command. The full sequence takes substantial GPU time.

## Scientific safeguards

- Quote searches use all 64 exported localized sites.
- Bracket audit searches use all 133 exported localized sites.
- Staged primary searches exclude only the explicitly frozen downstream handle and retain an all-133 audit.
- Known OpenAI site IDs are used only after selection for recovery reporting.
- `Dfit` builds signatures, `Dcal` calibrates top-`K` and strength, and `Dte` is final evaluation only.
- Failed signatures and failed redundancy searches remain in the report.
- The output task is binary; this release does not claim arbitrary-count bracket generation.

## Important directories

```text
artifacts/expected/   compact frozen evidence for the fast audit
configs/              exact experiment settings
data/                 complete 64-site and 133-site candidate CSVs
src/                  release CLI and experiment implementation
outputs/              generated only when rerunning experiments
```
