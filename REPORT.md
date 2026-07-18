# Ablation and Progressive Causal Discovery in OpenAI Sparse Circuits

## Technical summary

This experiment sequence produced one negative result and one strong positive result.

1. **The previously learned quote and bracket handles are important to the intact models.** Mean-clamping the quote handle `0.mlp.resid_delta:460` lowers quote accuracy from `1.000` to `0.542`. Mean-clamping the bracket binary-readout handle `4.attn.resid_delta:1079` lowers bracket accuracy from `1.000` to `0.589`.
2. **We did not find a behaviorally valid replacement after disabling either handle.** This is evidence against a readily available redundant implementation, but it is not proof that no redundancy exists.
3. **Progressive PLOT found a graded active-depth state upstream of the bracket binary readout.** The accepted high-level model on the tested distribution is

   ```text
   X -> D -> R -> Y
   ```

   where `D` is active bracket depth in `{1,2,3,4}`, `R = 1[D >= 2]`, and `Y` is the one-close versus two-close output.
4. **The discovered neural pathway closely matches OpenAI's manual circuit.** PLOT selected the upstream pair

   ```text
   2.attn.resid_delta:1249
   3.attn.resid_delta:1249
   ```

   which writes the graded depth signal read at `4.attn.act_in:1249`; the downstream threshold is `4.attn.resid_delta:1079`. OpenAI independently describes channel `1249` as list depth, layer 3 as amplifying layer 2's write to that channel, and channel `1079` as the thresholded nested-list decision.

The strongest current bracket interpretation is therefore:

```text
Prompt X
  -> graded active-depth signal D
       2.attn.resid_delta:1249
       3.attn.resid_delta:1249
  -> D read by layer 4
       4.attn.act_in:1249
  -> binary close decision R = 1[D >= 2]
       4.attn.resid_delta:1079
  -> output Y
```

This is stronger and more precise than `X -> R -> Y`. It does not certify an arbitrary bracket-generating algorithm: the evaluated output remains binary, one close versus two closes.

## Variables and running examples

### Quote

```text
X -> U -> Y
```

- `X`: the prompt.
- `U`: the type of the unmatched opening quote, single or double.
- `Y`: the matching closing quote.

Example:

```text
X = print("hello
U = double quote
Y = "
```

The certified neural handle for `U` is `0.mlp.resid_delta:460`.

### Bracket

```text
X -> D -> R -> Y
```

- `D`: active bracket depth, tested at depths `1`, `2`, `3`, and `4`.
- `R`: the binary decision `R = 0` when `D = 1`, and `R = 1` when `D >= 2`.
- `Y`: `]` when `R = 0`, and `]]` when `R = 1`.

Examples:

```text
D = 1  -> R = 0 -> Y = ]
D = 2  -> R = 1 -> Y = ]]
D = 4  -> R = 1 -> Y = ]]
```

The comparison between `D = 2` and `D = 4` is essential. It changes `D` while preserving `R` and `Y`; this is what lets us distinguish a graded depth variable from a binary output proxy.

## Candidate universes and data separation

| Task | Model | Candidate sites | Filtering | Candidate CSV SHA256 |
|---|---|---:|---|---|
| Quote | `csp_yolo1` | 64 | none for the necessity audit; only the explicitly ablated handle removed for rediscovery | `c38db7b63313960577c6b214f3bdb8979d126532b7afe5ee1f853f6cdf2ae01a` |
| Bracket | `csp_yolo2` | 133 | none for audit searches; only a frozen downstream handle removed in a staged primary search | `4379a582f1d57051e5e8ebbf7e84252c738bd6708da4646e08f7e23d967be547` |

Every PLOT search used singleton neural candidates over the complete localized circuit. Top-`K` calibration could combine ranked singletons into a multi-site executable handle. Known OpenAI sites were not used to construct rankings; they were checked only after selection.

The banks were split by content:

- `Dfit`: signatures and any frozen scalar readout.
- `Dcal`: top-`K` and intervention-strength selection.
- `Dte`: one final heldout evaluation only.

The manifests assert that the content sets are disjoint. Unit tests enforce that `Dte` cannot affect ranking or calibration.

## Method

### Frozen-handle necessity

For each neural site, we computed its unconditional mean over all tokens in the task-specific `Dfit` bank. A global ablation replaced the selected activation by this fixed value at every position:

```text
h_s(x, p) <- mean_Dfit,tokens(h_s)
```

We measured:

- binary task accuracy;
- signed output-margin drop;
- binary contrast-loss increase;
- next-token NLL increase.

All localized singleton sites were ablated independently to rank each learned handle against the full circuit. Confidence intervals resample content clusters, not individual prompts.

This mean is a task-bank mean, not OpenAI's exact pretraining-distribution mean. Consequently, the full-model damage measurements are useful, but the attempted circuit-only reconstruction is not an exact replication of OpenAI's pruning baseline.

### Ablate and rediscover

We continuously clamped the learned handle to its task mean, then reran PLOT over every remaining localized singleton.

For each source/base pair and candidate site `s`, the patch was

```text
h_s(swap) = h_s(base) + lambda * w_s * (h_s(source) - h_s(base)).
```

The abstract effect was the source-minus-base change in the causal variable. The neural effect was the canonical output intervention effect:

```text
phi(y_swap_s) - phi(y_base),
```

where `phi` is the positive-class logit minus the negative-class logit. Dfit matching used raw cosine-cost one-sided UOT with `epsilon = 0.08` and `beta = 0.08`; Dcal selected top-`K` and strength.

### Progressive bracket search

The staged search used an already validated downstream handle to define the next neural effect, while preserving the same source/base patch semantics.

1. Freeze the binary readout `R` at `4.attn.resid_delta:1079`.
2. Fit only on Dfit the sign and threshold that decode its two states.
3. For every candidate `s`, use

   ```text
   decoded_R(y_swap_s) - decoded_R(y_base)
   ```

   as the neural effect signature and `R(source) - R(base)` as the abstract signature.
4. Calibrate the selected support on Dcal and validate both internal-state movement and output behavior on Dte.
5. Restore `R` to its base value after the upstream patch. If this restores the base output, the upstream site's output effect is mediated by `R`.

The selected upstream site moved continuously between depths `2`, `3`, and `4`, even though `R` stayed fixed. That observation motivated a new, preregistered active-depth experiment on a fresh bank.

### Graded active-depth search

For the sealed follow-up, the abstract variable was explicitly

```text
D(x) = active bracket depth in {1,2,3,4}.
```

The bank crossed depth with all four context families and neutral bracket-free padding counts `q in {0,1,2,4}`. The model's clean accuracy was `1.000` on Dfit, Dcal, and Dte.

An affine decoder from `4.attn.act_in:1249` to `D` was fit on Dfit only and frozen. The direct intervention tested whether patching this site transfers the exact source depth while preserving the binary readout and output whenever both prompts remain on the same side of the `D >= 2` threshold.

The upstream PLOT search then excluded only this frozen downstream depth handle from its 132-site primary universe. For every remaining candidate `s`:

```text
abstract signature = D(source) - D(base)
neural signature   = decoded_D(y_swap_s) - decoded_D(y_base).
```

An all-133-site audit was run in parallel as a sanity check. The frozen depth handle ranked first there.

## Result 1: the learned handles are important

### Quote handle

| Quantity | Result |
|---|---:|
| Intact accuracy | 1.000 |
| Accuracy after global `U460` ablation | 0.542 |
| Accuracy drop, 95% cluster bootstrap CI | `[0.432, 0.484]` |
| Mean signed-margin drop | 4.079 |
| Margin-drop 95% CI | `[3.959, 4.203]` |
| Rank among 64 singletons by accuracy damage | 1 |
| Rank by margin damage | 1 |
| Rank by contrast-loss damage | 1 |

The quote handle is not merely sufficient under patching: removing it from the intact model causes the largest binary-accuracy, margin, and contrast-loss damage among all 64 localized sites.

### Bracket handles

| Ablated handle | Accuracy after ablation | Accuracy drop | Margin drop | Accuracy rank / 133 | Margin rank / 133 |
|---|---:|---:|---:|---:|---:|
| `R_mid = 4.attn.resid_delta:1079` | 0.589 | 0.411 | 2.961 | 3 | 1 |
| Published depth comparator `2.attn.resid_delta:1249` | 0.875 | 0.125 | 1.953 | 13 | 5 |
| Three-site prior coarse `R` handle | 0.464 | 0.536 | 2.834 | set-level | set-level |
| All 133 localized sites | 0.250 | 0.750 | 3.508 | set-level | set-level |

For `R_mid`, the 95% cluster-bootstrap interval for the accuracy drop is `[0.383, 0.440]`; the margin-drop interval is `[2.924, 2.998]`. This is strong necessity evidence for the binary threshold site.

### Sufficiency caveat

Keeping only the exported localized nodes while mean-clamping every other exposed node produced clean accuracy `0.536` for quote and `0.542` for bracket. Both fail the preregistered `0.90` gate, so no circuit-only ablation result is interpreted.

The likely issue is methodological mismatch: OpenAI uses means over the pretraining distribution, whereas this run had only task-bank means. This does not invalidate the intact-model singleton ablations, but it prevents a necessity-and-sufficiency claim from this audit alone.

## Result 2: no replacement was recovered after ablation

| Task | Disabled site | Best remaining calibrated handle | Clamped clean Dte accuracy | Dte sensitivity | Dte invariance summary | Pass? |
|---|---|---|---:|---:|---:|---|
| Quote | `0.mlp.resid_delta:460` | `0.mlp.post_act:2790 + 0.mlp.act_in:180` | 0.542 | 0.708 | 0.500 same-variable preserve | no |
| Bracket | `4.attn.resid_delta:1079` | `7.mlp.post_act:4133` | 0.646 | 0.771 different-`R` flip | worst required preserve 0.813 | no |

The result is deliberately stated narrowly:

> Under continuous mean-clamping of the certified handle, PLOT did not find a compact remaining-site intervention that restored the required heldout behavior.

Because the clamped models themselves fall well below `0.90` clean accuracy, this is not evidence that the original model has no distributed or conditional redundancy. It does show that there is no readily recoverable alternate handle in the remaining localized sites under this intervention.

## Result 3: progressive PLOT found an upstream graded state

### Transparent account of the unsuccessful intermediate runs

Two initial signatures failed and are retained as negative evidence:

1. **Raw `R_mid` activation plus output effect.** This selected the downstream site `7.mlp.act_in:1079`; heldout score was `0.726`, and mediation failed. The output block encouraged relocalization toward output rather than upstream discovery.
2. **Raw `R_mid` activation only.** This selected `3.attn.v:1385` and failed. Inspection showed that the raw `R_mid` scalar had the opposite sign from the abstract `R` convention. The certified `R_mid` site ranked last in the all-site sanity check, exposing the sign error.

The correction was principled: fit the binary orientation and threshold on Dfit, freeze them, and use the decoded downstream variable. No Dte result was used to make this correction.

### Corrected staged result

The corrected all-site search selected:

```text
4.attn.act_in:1249
K = 1
lambda = 1.0
```

All required Dcal and Dte state/output rates were `1.000`. On different-`R` heldout pairs:

- patching the site moved decoded `R_mid` to the source state: `1.000`;
- output changed to the source answer: `1.000`;
- restoring `R_mid` to base restored the base output: `1.000`;
- mean removed output-effect fraction: `0.858`.

However, the scalar activation also moved on same-`R`, different-depth pairs. Therefore the correct interpretation is not a second binary variable. It is a graded quantity upstream of the binary threshold.

### Choosing the graded variable

We did not immediately relabel the selected site as depth. We first compared several possible graded quantities using exploratory Dfit-only diagnostics:

| Candidate quantity | Relationship with `4.attn.act_in:1249` |
|---|---:|
| Active bracket depth | Pearson `0.997` |
| Binary `R` | Pearson `0.785` |
| Total surface open brackets | Pearson `0.542` |
| Prompt length | Pearson `-0.066` |
| Active-depth density | Pearson `0.606` |
| Surface-bracket density | Pearson `0.411` |

Within fixed context, content, and padding strata, the mean correlation with active depth was `0.9996` and the minimum was `0.9980`. Separate pilot runs for surface density and active-depth density achieved only about `0.60` and `0.75` heldout Pearson correlation, respectively, so neither was promoted.

These diagnostics were used only to select the next abstract-variable hypothesis. The active-depth claim below was then tested on a newly generated, sealed Dfit/Dcal/Dte bank.

## Result 4: active depth is validated on fresh heldout data

### Representation quality

| Split | Pearson correlation with active depth | R2 | MAE in depth units |
|---|---:|---:|---:|
| Dfit | 0.9967 | 0.9934 | 0.0700 |
| Dcal | 0.9971 | 0.9941 | 0.0682 |
| Dte | 0.9969 | 0.9936 | 0.0710 |

Thresholding the decoded depth at `D >= 2` predicts the frozen `R_mid` state with `1.000` accuracy on all three splits.

### Direct causal validation of `4.attn.act_in:1249`

Every heldout aggregate below passed at `1.000`:

| Heldout relation | Exact source depth | Correct `R_mid` | Correct output |
|---|---:|---:|---:|
| Different depth, different `R` | 1.000 | 1.000 | 1.000 |
| Different depth, same `R` | 1.000 | 1.000 preserve | 1.000 preserve |
| Same depth, different neutral padding | 1.000 | 1.000 preserve | 1.000 preserve |
| Wrong numeric content | 1.000 | 1.000 preserve | 1.000 preserve |

For different-`R` pairs, restoring `R_mid` to base restored base output at `1.000`, and removed `86.5%` of the mean output effect. Thus the data support `D -> R_mid -> Y`, not merely correlation between `D` and output.

### Progressive discovery upstream of the frozen depth handle

The all-133-site audit ranked:

| Audit rank | Site | Interpretation |
|---:|---|---|
| 1 | `4.attn.act_in:1249` | frozen downstream depth read; sanity check |
| 2 | `2.attn.resid_delta:1249` | upstream graded-depth write |
| 3 | `3.attn.resid_delta:1249` | layer-3 amplification/copy of channel 1249 |

After excluding only the frozen site, the strict calibrated primary handle was:

```text
K = 2, lambda = 2.0
0.502136 * 2.attn.resid_delta:1249
0.497864 * 3.attn.resid_delta:1249
```

All Dcal and Dte aggregate gates were `1.000`. The upstream patch transferred the exact decoded source depth while preserving `R_mid` and output on same-side transitions. On threshold-crossing transitions, it moved both `R_mid` and output to the source state at `1.000`. Restoring the downstream depth handle to base restored output at `1.000`; mean removed output-effect fraction was `0.744`.

The known OpenAI site was not supplied to selection. It was primary rank `1` and included in the strict selected pair.

## Agreement with OpenAI's manual circuit

The match is stronger than a shared coordinate number:

1. OpenAI says layer-2 attention writes an averaged open-bracket detector into residual channel `1249`, whose magnitude represents list depth.
2. OpenAI says layer-3 attention also writes channel `1249`, highly correlates with the layer-2 write, and appears to amplify it.
3. OpenAI says layer-4 attention reads list depth and thresholds it, writing the nested-list decision into residual channel `1079`.
4. The staged PLOT ranking did not use these known site IDs. It selected exactly the layer-2 and layer-3 `1249` writes, then validated their causal effect through the layer-4 `1249` read and the `1079` threshold.

This provides a causal abstraction for the two paper sites:

```text
X -> D -> R -> Y
```

The multiple `1249` neural sites are sequential writes/reads of the same abstract variable `D`; they should not be counted as separate high-level variables without additional evidence.

## What is and is not established

### Established on the tested distribution

- The quote handle `0.mlp.resid_delta:460` is strongly important to the intact quote model.
- The bracket threshold handle `4.attn.resid_delta:1079` is strongly important to the intact bracket model.
- `4.attn.act_in:1249` carries a graded active-depth state that can be patched independently of the binary output decision.
- The upstream pair `2.attn.resid_delta:1249 + 3.attn.resid_delta:1249` causally transfers that graded state.
- The graded state's output effect is substantially mediated by the binary `1079` state.
- The refined bracket abstraction `X -> D -> R -> Y` passes all preregistered heldout gates in this run.

### Not established

- We have not shown that the circuit can emit an arbitrary number of closing brackets. `Y` is still binary.
- We have not proven that `D` remains an exact absolute-depth variable under arbitrarily long contexts. OpenAI reports context dilution because the upstream attention computes an average.
- We have not certified a separate later high-level variable `R_late` in this experiment sequence.
- We have not proven absence of redundancy; only failure to recover an alternate handle under continuous ablation.
- We have not reproduced OpenAI's circuit-only sufficiency result because their exact pretraining means were unavailable.
- The final depth experiment uses one deterministic content bank. It has fresh Dcal/Dte separation, but not yet repeated-bank uncertainty estimates.

## Recommended next experiments

### 1. Repeat and bootstrap the graded-depth discovery

This is the highest-priority confirmation. Repeat the exact frozen design across several new content offsets and bootstrap Dfit/Dcal records. Report:

- rank stability of the two `1249` writes;
- frequency with which the strict calibrated handle is `K=1` versus the layer-2/layer-3 pair;
- heldout pass frequency;
- confidence intervals for mediation fractions.

Do not redesign the signature during this robustness pass.

### 2. Map the abstraction boundary under long-context dilution

Vary bracket-free context length far beyond `q = 4`, while holding active depth fixed. Test competing preregistered variables:

```text
D_absolute = active depth
D_effective = active open-bracket evidence / context length
```

Use separate PLOT rows and sealed test data. This determines whether `D` is an exact causal abstraction or a task-local approximation to the averaged signal described by OpenAI.

### 3. Obtain or reconstruct the correct pretraining means

Search the released OpenAI artifacts for stored activation means. If absent, estimate them on a pinned sample from the original tokenizer/pretraining distribution. Then rerun:

- circuit-only sufficiency;
- singleton necessity;
- ablate-and-rediscover.

This is required for a direct apples-to-apples comparison with OpenAI's pruning claims.

### 4. Test the threshold edge more directly

Freeze the selected depth handle and intervene on the layer-4 attention selector/message that maps channel `1249` into channel `1079`. The goal is to distinguish:

```text
D representation -> threshold operation -> R
```

from a less specific shared-path explanation. Predeclare selector-only, value-only, message-patch, and message-block interventions.

### 5. Extend the output task only after the current model is robust

Construct a new task requiring `1`, `2`, `3`, or `4` closing brackets. The present model predicts only a binary output and cannot by itself justify arbitrary-count closure. A multi-output task would test whether the graded `D` state can drive a richer decoder.

## Reproducibility

Primary result roots:

```text
eval/openai_sparse_plot/frozen_handle_necessity_20260715
eval/openai_sparse_plot/ablate_rediscover_20260715
eval/openai_sparse_plot/progressive_rearly_decoded_20260715
eval/openai_sparse_plot/graded_depth_20260715
```

Implemented runners:

```text
experiments/openai_sparse_plot/run_frozen_handle_necessity_audit.py
experiments/openai_sparse_plot/run_ablate_and_rediscover.py
experiments/openai_sparse_plot/run_progressive_rearly.py
experiments/openai_sparse_plot/run_graded_evidence_plot.py
```

The final graded-depth run used:

```powershell
$env:PYTHONPATH="$PWD\.venv_sparse_plot\Lib\site-packages;$PWD\.external\circuit_sparsity;$PWD"

python -m experiments.openai_sparse_plot.run_graded_evidence_plot `
  --circuit-home .external\circuit_sparsity `
  --candidate-csv eval\openai_sparse_plot\bracket_counting_inventory_csp_yolo2_prune_v4\string_closing_circuit_nodes.csv `
  --expected-node-count 133 `
  --parent-rearly eval\openai_sparse_plot\progressive_rearly_decoded_20260715\progressive_rearly.json `
  --out-dir eval\openai_sparse_plot\graded_depth_20260715 `
  --e-definition active_depth `
  --content-offset 18000 `
  --q-grid 0,1,2,4 `
  --fit-contents 16 `
  --cal-contents 8 `
  --test-contents 8 `
  --fit-records-per-relation 64 `
  --cal-records-per-relation 48 `
  --test-records-per-relation 64 `
  --k-grid 1,2,3,5,8 `
  --strength-grid 0.5,1.0,2.0,4.0 `
  --selector-epsilon 0.08 `
  --selector-beta 0.08 `
  --cuda
```

Verification completed on 2026-07-15:

```text
py_compile: passed
unittest discover experiments/openai_sparse_plot/tests: 226 tests passed
```

Reference: [Gao et al., *Weight-sparse transformers have interpretable circuits*](https://arxiv.org/pdf/2511.13653).
