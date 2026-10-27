# Full Qwen3 TRL floating-point discrepancy diagnosis

- Status: bounded diagnostic evidence; original strict failures remain open.
- Reviewed: 2026-10-22.
- Baseline: `8d64cd1c9d5b135fcaa4eef7ba04dad7600f9f15`; synchronized target
  `2.0-refactor` at `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`.
- Owner: numerical kernels and Torch compatibility maintainers.
- Recheck when: Qwen/TRL sources, numerical kernels, scalar promotion, precision
  settings, or the tested checkpoint changes.

## Conclusion

The controlled experiments strongly support FP32 rounding and operation-order
sensitivity as the main explanation for the observed discrepancy. They do not
prove that every failed element across all ten DPO/GRPO updates is benign, and
do not turn the original strict failures into passes. No runtime implementation
or numerical acceptance threshold was changed by this investigation.

The first differing module is the first block's RMSNorm: its inputs and weights
are identical, and its FP32 outputs differ by at most **3 ULP**. With explicit
FP64 constants and internal computation, a complete 28-layer DPO forward reduces
cross-runtime maximum logit error from **7.10487e-5 to 1.17240e-13**.

There is a separate, real FP64 compatibility issue, already tracked as
**KI-DTYPE-003**: Python floating-point scalars are first narrowed to FP32.
This is not normal FP64 rounding. The high-precision diagnostic uses explicit
FP64 tensor constants to avoid contaminating the control. Original FP32 training
does not demand double-precision scalar arithmetic.

## Scope and controls

Same full Qwen3-0.6B checkpoint, **596,049,920 parameters**, 28 blocks, tied
embedding/head, FP32 eager attention, no LoRA, no optimizer update. Independent
Torch 2.8.0+cu128 and Jittor interpreters use Transformers 4.56.2, TRL 0.21.0 and
NumPy 1.26.4. Hardware and artifact pins are those in the
[full training report](2026-10-21-trl-dpo-grpo-qwen3.md).

- Read back and hash all initial parameters and buffers against the completed
  archived training trace. Both runtimes use exactly the same archived CUDA
  step-zero batch and reference probabilities, including for CPU controls.
- DPO replays one chosen/rejected pair, `[2,13]` tokens. GRPO replays its archived
  first group, `[2,30]` tokens with 16 completion positions; advantages and
  reference probabilities are fixed. No generation, reward scoring or training
  trajectory is rerun here.
- Use the original Qwen forward for FP32, TRL `selective_log_softmax` and the
  corresponding DPO/GRPO loss formula. Each genuine Torch CUDA replay matches
  all **310 observed parameter-gradient fields** from the official Trainer's
  first step **exactly**: full tensors where retained, deterministic samples
  for the other large tensors.
- Observe embedding, Linear, RMSNorm and block outputs; collect full gradients
  for all norms and blocks 0/27, adding block 2 in the GRPO/second diagnostic
  revision. Sample every 4096th element of the other parameter gradients.
  This is not a claim to retain all 596 million gradient elements.
- Every diagnostic completes with zero backend fallback. TF32 is disabled and
  CUDA kernel math is strict. CPU hides CUDA. Physical observations are checked.
- Diagnostic FP64 replaces Qwen's explicit FP32 RMSNorm, RoPE and attention
  softmax casts. Explicit FP64 epsilon, attention scale and loss-beta constants
  avoid KI-DTYPE-003. Default RoPE and zero dropout are applicable to this model.
  These process-local diagnostic overrides do not change the library or its
  reported production FP32 behavior.

## DPO same-weight forward

Maximum absolute differences across the full `[2,13,151936]` logit tensor:

| Comparison | Max absolute error | Relative L2 |
| --- | ---: | ---: |
| Jittor vs Torch, CPU FP32 | 7.10487e-5 | 2.71553e-6 |
| Jittor vs Torch, CUDA FP32 | 3.29018e-5 | 1.14052e-6 |
| Torch CPU vs Torch CUDA, FP32 | 7.83205e-5 | 2.52641e-6 |
| Jittor vs Torch, CPU explicit FP64 | 1.17240e-13 | 3.77694e-15 |

Thus the cross-framework FP32 discrepancy is comparable to PyTorch's own
cross-device difference under this controlled input. The FP64 experiment
reduces maximum discrepancy by approximately **6.1e8 times**. This supports a
finite-precision explanation; CPU/GPU variation alone would not prove it.
PyTorch's own [numerical-accuracy documentation](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)
explains why operation order, batching and device can change floating-point
results even for identical inputs.

Both FP32 implementations also remain similarly close to the explicit Torch
FP64 reference: CPU logit maxima are 6.01882e-5 (Torch) and 7.27057e-5 (Jittor);
CUDA maxima are 2.70093e-5 and 2.68764e-5 respectively. PyTorch FP32 is not an
exact-real-arithmetic ground truth.

## First RMSNorm and fusion ablation

An independent NumPy FP64 calculation with the actual embedding output and
checkpoint norm weight confirms:

- First-norm inputs are bit-identical. Jittor/Torch FP32 differ by at most
  9.53674e-7, 3 ULP; the 99th percentile is 2 ULP on CPU and CUDA.
- Against the FP64 same-input norm, Torch FP32 maximum error is 2.89935e-7 and
  Jittor FP32 is 9.40807e-7. Both are within 2–3 ULP of the FP64 reference
  rounded to FP32.
- Disabling only Jittor's automatic fused RMSNorm substitution on CUDA gives
  norm error 2.89935e-7 against that reference. Full-model logit discrepancy
  changes from 3.29018e-5 to 3.14713e-5; relative L2 actually rises from
  1.14052e-6 to 1.21814e-6. This is **not** a general accuracy fix or evidence
  that RMSNorm alone explains the full error.

The initial FP64 experiment retained ordinary Python scalar constants. Its
first-norm error was 6.68976e-11 against exact epsilon `1e-6`, but only
8.88178e-16 against `float(np.float32(1e-6))`. This isolates scalar narrowing.
The explicit-constant FP64 run has exactly matching first-norm outputs. The
initial, scalar-contaminated FP64 backward diagnostic still reduces
selected-gradient maximum discrepancy from 1.08331e-5 to 3.48642e-8 and relative
L2 from about 1.29e-5 to 8.17e-8; it is not presented as a pure FP64 control.
The corrected explicit-FP64 Jittor experiment observes the forward only.

## GRPO gradient hotspot

The original GRPO CUDA first update already failed one element of
`model.layers.2.mlp.down_proj.weight`; maximum tensor discrepancy was
1.67266e-5. Therefore optimizer drift cannot explain every original failure.

Replaying that group with a **shared** reference-probability array gives a
maximum difference of 1.51545e-5 in this matrix, with no element above its
unchanged `1e-5 + 1e-4 * scale` budget. Against the explicit Torch FP64 reference:

| FP32 runtime | Matrix max absolute gradient error | Relative L2 |
| --- | ---: | ---: |
| Torch CPU | 2.00809e-5 | 8.41353e-6 |
| Jittor CPU | 3.04553e-05 | 1.18486e-05 |
| Torch CUDA | 1.86741e-5 | 1.04048e-5 |
| Jittor CUDA | 1.69926e-5 | 9.54444e-6 |

Jittor's hotspot deviation from higher precision is comparable to Torch's.
The common-input CPU replay still fails 20 elements across the observed full
gradient fields; CUDA fails none. Against FP64, those fields contain 2 failed
elements for Torch CPU, 11 for Jittor CPU, 6 for Torch CUDA and 1 for Jittor CUDA.
Even PyTorch FP32 versus its own FP64 control can exceed this per-element budget;
these counts are diagnostics, not a reason to change the original gate.
However, the original Jittor training used its own reference forward, and the
module snapshots may alter lazy fusion boundaries. Consequently this replay
is a diagnostic, **not a reclassification of the archived one-element failure**.

## Reproduce and inspect

Reusable tools:

- `agent/skills/torch-compat-repo-runbook/scripts/trl_rounding_probe.py`
- `agent/skills/torch-compat-repo-runbook/scripts/trl_rounding_compare.py`

Use the existing independent environments and offline CPU/CUDA configuration
from the runbook. For each runtime/precision/device, use a new output directory:

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/trl_rounding_probe.py \
  --root "$PWD" --runtime torch --device cpu --precision fp64 --backward \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --batch-trace "$JITTOR_LAB_ROOT/_state/trl-rl-qwen3/dpo-torch-cuda-10/trace" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/torch-fp64"
```

Set `--runtime jittor`, `--precision fp32|fp64`, `--device cpu|cuda` as needed;
use `--algorithm grpo` with the archived GRPO CUDA oracle trace for GRPO.
`--eager-rms` disables the Jittor fused-norm shortcut for a diagnostic ablation.
Omit `--backward` for the explicit-FP64 Jittor forward control measured here.

```bash
<analysis-python> agent/skills/torch-compat-repo-runbook/scripts/trl_rounding_compare.py \
  --reference "$JITTOR_LAB_ROOT/_state/<new-run>/torch-fp64" \
  --candidate "$JITTOR_LAB_ROOT/_state/<new-run>/jittor-fp64" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/comparison.json"
```

The comparer verifies every saved array hash, identical initial state and batch
provenance, complete zero-fallback runs and actual FP64 observation dtypes.
Full-gradient and sampled-gradient statistics are separate; missing observation
keys are explicit. Different precisions/devices are intentional controls.

Unversioned evidence is under `$JITTOR_LAB_ROOT/_state/trl-rounding/`: independent
run manifests/arrays, `audited-comparisons.json`, exact archived diagnostic source
revisions `measured-v1.py` and `measured-v2.py`, official-step-zero control receipts,
launch configuration, `first-norm-audit.json` and logs. The original FP32 runs used revision 1; revision
2 adds explicit FP64 constants, GRPO, the fusion ablation and full block-2 gradients.
Both measured revisions remain available; hashes are recorded in each manifest.

## Remaining limits

No ten-step trajectory was repeated in FP64, no proof is claimed for every
original failed gradient/logit element, and no optimizer or training-quality
conclusion follows from these one-step controls. The unchanged full-model
strict-budget failure remains KI-COMPAT-017. FP64 scalar narrowing remains
KI-DTYPE-003. This diagnosis does not establish a universally appropriate
acceptance threshold, convergence, throughput, or whole-library compatibility.

## Tool validation

Fourteen diagnostic runs completed with zero fallback; sixteen pairwise audits
verified all saved payload hashes and FP64 dtypes. Both archived source
revisions and all process exit codes were checked. The new comparison controls
plus existing trace/logit utility tests passed **54 tests**; Ruff, repository
layout, generated-manifest consistency and whitespace checks passed. Runtime
kernels were unchanged. The required full structure suite passed **1516 tests**,
with **4 skips**.
