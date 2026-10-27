# Local TRL fixes: SGD memory, AMP boundaries and retained evidence

- Reviewed: 2026-10-24.
- Owner: Torch compatibility and downstream validation maintainers.
- Recheck when: upstream pins, optimizer state, AMP/scalar arithmetic, serialization or the measured hardware changes.
- Status: bounded fixes verified; strict training parity and CUDA AMP continuation remain unresolved.
- Synchronized upstream baseline: `ca56aa802c2acf9a22bca32d1bee3276381ed203`, merged locally in `33273e5d5e7bc568e47c5d54b2ec0a8da23a59f2`.
- Scope: the two previously failing full Qwen3-0.6B CUDA SGD cases, practical SFT LoRA/AdamW AMP controls, and evidence restoration. This is not whole-TRL support or L2 admission. The upstream trainer breadth remains the previously selected 11 of 18 trainers, not a new whole-library denominator.

This extends the [previous validation report](2026-10-23-trl-full-and-practical.md). Pins remain TRL 0.21.0, Transformers 4.56.2, Accelerate 1.10.1, datasets 3.6.0, PEFT 0.17.1 for practical SFT, and genuine Torch 2.8.0+cu128. The full checkpoint and shared initialization artifacts remain unchanged. Cross-runtime floating-point comparisons retain `atol=1e-5, rtol=1e-4` with the existing field-scale floor; controlled inputs and own-runtime continuation use exact comparisons. No tolerance was loosened.

Receipts below are relative to `$JITTOR_LAB_ROOT/_state/trl-local-fixes/` unless stated otherwise. Launch receipts bind commands, environment, commit and, where applicable, the measured working-tree patch/hash. Several measurements preceded their final commits; a recorded commit alone does not describe those measured sources.

## Changes and verification

| Local change | Result and boundary |
| --- | --- |
| `e947beb68` — optional SGD momentum state | Momentum-free SGD no longer materializes unused full-sized momentum state. Momentum activation and checkpoint loading preserve optional-state behavior. The two previously blocked CUDA cases now finish. |
| `9902da732` — lossless reference restoration | Archived oracle arrays can be restored into a new verified destination without changing their source archives or original manifests. This supplies the raw references required by the frozen producer. |
| `df3d3899b` — Torch autocast boundaries | Preserves the measured FP32 arithmetic boundaries and low-precision contraction behavior, including convolution interfaces. Focused controls pass within their stated device exclusions; full AMP training remains numerically different. |
| `87dea3e39` — CUDA scalar multiplication | FP16/BF16 tensors retain the FP32 Python-float coefficient through multiplication. Primitive controls pass; full training and continuation gaps remain. |
| `470c48688` — corrected counter evidence | Corrects the diagnosis, not the optimizer implementation: live Adam step state is a Python float; the original saved checkpoint counters were already FP32 tensors. The live-state contract remains open. |

Focused checks and their logs:

| Check | Result | Receipt/log |
| --- | --- | --- |
| CPU optional-state/checkpoint cases | 86 passed | `optional-state-after-r2.log` |
| CUDA native SGD, policy checkpoint, fused SGD and step capture | 40 passed | `sgd-after-cuda.json`, `.log` |
| CUDA Torch SGD execution options, momentum state and continuation | 76 passed | `sgd-compat-final-cuda.json`, `.log` |
| AMP boundary, fidelity and training-loop tests on CPU | 81 passed | `amp2-final-suite-cpu.json`, `.log` |
| Same AMP tests on CUDA | 80 passed, one explicitly deselected | `amp2-final-suite-cuda.json`, `.log` |
| Lossless archive/restore tools | 46 passed | `archive-final-tests.log` |
| Scalar-fix CPU neighboring tests | 57 passed | `amp4-nearby-bounded-cpu.log` |
| Scalar-fix CPU FP32 controls | One test / 48 subcases passed, independently matched by Torch | `amp4-scalar-final-cpu.log`, `amp4-scalar-oracle-final-cpu.log` |
| Scalar-fix CUDA neighboring tests and scalar controls | 57 passed, two deselected; includes 144 scalar subcases independently matched by Torch | `amp4-nearby-bounded-cuda.log`, `amp4-scalar-oracle-final-cuda.log` |
| Nonambient CUDA placement | One test / 32 subcases passed with two visible GPUs | `amp4-scalar-two-gpu.log` |
| Final Torch-mode CPU core | 224 passed, one skipped | `core-scalarfix-cpu-final.json`, `.log` |
| Final documentation/path/deferred contracts | 22 passed | `docs-local-fixes-final.log` |
| Final repository structure | 1,516 passed, four skipped | `structure-scalarfix-final.json`, `.log` |

These are recorded checks of their source-bound revisions, not a claim that every subsequent working-tree change has received the same complete suite. CUDA AMP's excluded case is `test_fp32_linear_and_convolution_bias_keep_bfloat16`; its underlying route failure is documented below rather than counted as a pass. Commands are preserved in the JSON receipts; the maintained tests include `tests/optim/test_sgd_lazy_momentum.py`, `tests/optim/test_sgd_policy_checkpoint.py`, `tests/optim/test_fused_sgd_cuda.py`, and the Torch compatibility SGD/AMP suites.

The final scalar suite's two CUDA deselections are the native BF16 convolution
case above and the nonambient-device case tested separately with two visible
GPUs. CPU low-precision scalar parity is not claimed. Ruff reports zero new
violations against the source baseline. Layout checks and generated manifest
checks pass.

An expanded exploratory CPU run is retained as a nonpass:
`amp4-nearby-final-cpu.log` has 70 passes and a
`TestFastViews.test_permutes` producer-node assertion failure. The broader CUDA
fast-path run was interrupted while compiling unrelated indexing kernels;
`amp4-scalar-final-cuda.log` is not accepted as a completed pass. The fresh,
bounded suites in the table explicitly replace those exploratory runs for the
changed multiplication/AMP scope; no full fast-path-suite pass is claimed.

## Full-model CUDA SGD results

Online DPO and RLOO each complete ten optimizer calls with zero backend fallback after the SGD fix, against the independently measured genuine Torch references. These are full 28-layer pretrained models with all policy parameters trainable. Both still fail the unchanged aggregate strict numerical test.

| Method | Final gradient max abs | Final gradient failing tensors | Next-token logits max abs | Next-token relative L2 | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Online DPO | 0.000472128 | 153/310 | 0.000800133 | 2.35883e-5 | 10/10 |
| RLOO | 0 | 0/310 | 0.0111647 | 3.50924e-4 | 10/10 |

The logits cover the complete vocabulary at the final position of ten fixed prompts. Mean Torch-to-Jittor softmax KL is `4.20960e-10` for Online DPO and `3.08116e-7` for RLOO. Matching these ten top-1 choices is not a model-quality result.

RLOO's final parameter gradients are zero in both runtimes, so final-gradient relative L2 is undefined. Ten optimizer calls do not mean ten nonzero updates. RLOO still has no directly recorded scalar training-loss comparison. Final parameter values pass the budget in both cases, but that does not establish equality relative to the much smaller training update.

The evidence retains intermediate gradient/parameter samples and complete final-step gradients and parameter values. Candidate full arrays were stream-compared, with their hashes/statistics retained rather than duplicate bulk payloads; they cannot be independently recomputed from candidate bytes afterward. The restored oracle arrays remain independently readable. Relevant receipts are `online_dpo-cuda-sgdfix-10-r2-audit.json`, `rloo-cuda-sgdfix-10-r2-audit.json` and their launch/source-patch records. This changes the earlier CUDA execution outcome; it does not erase the original OOM or numerical failures.

## AMP SFT after the boundary fix, before the scalar follow-up

Fresh genuine Torch and Jittor runs used the same physical CUDA device, the same pinned input/model/adapter artifacts and source-matched producer. The full base stays frozen; LoRA trains 1,146,880 parameters across 112 tensors with AdamW. Both precisions finish ten actual updates with zero fallback. FP16 uses the explicitly shared initial GradScaler scale of 1024. Each runtime also restores its own checkpoint 5 in a new process and performs steps 6–10.

| Profile | Next-token logits max abs | Final adapter error L2 / Torch update L2 | Jittor exact resume | Resume logits max abs |
| --- | ---: | ---: | --- | ---: |
| FP16 | 0.328125 | 4.468680872% | Fail | 0.1875 |
| BF16 | 2.296875 | 14.438064870% | Fail | 1.5625 |

Both cross-runtime comparisons fail strict parity. Genuine Torch's own continuation is exact in both profiles. The three fixed next-token top-1 choices agree, but the distributions and continuation are not equivalent. Adapter-update error decreases from the earlier 5.181812857% / 15.872279373%; this is a narrow metric improvement, not proof that overall AMP parity improved. In particular, BF16 next-token error is larger than the previous report's 1.498046875.

Use `ampfix-practical-run-summary.json`, `ampfix-{fp16,bf16}-cross-comparison.json`, `ampfix-*-adapter-update-error.json`, `ampfix-*-final-logits-diagnostic.json` and the four `ampfix-*-{torch,jittor}-resume-comparison.json` receipts. Run manifests bind the inputs, producer sources, actual update steps and checkpoint provenance. These rows precede the scalar change; they must not be relabeled as its validation.

### CUDA Python-scalar multiplication: repaired and rechecked

`amp3-scalar-narrowing-proof.json` isolates another arithmetic gap: CUDA
FP16/BF16 Tensor multiplication narrows `1/sqrt(128)` to `0.08837890625`
before multiplication, while genuine Torch uses the FP32 coefficient
`0.0883883461356163` and rounds the output once. This affects Qwen3 eager
attention's `matmul(...) * scaling`, both with and without autocast. SiLU,
module SiLU and Qwen's no-bias linear controls did not reproduce that gap.

The compatibility frontend now composes existing native FP32 arithmetic and
one output cast. It handles builtin Python floats with CUDA FP16/BF16 tensors
for `*`, reverse `*`, Tensor `mul`/`multiply` and module-level `mul`/`multiply`.
It preserves the input device, output dtype and gradient. It does not change
native Jittor arithmetic, CPU low-precision policy, integer scalars, in-place
multiplication or the existing `out=` delegate boundary. Focused independent
CUDA controls cover positive/negative non-dyadic scalars and zero, both autocast
states, eight spellings and the FP32 fast path. A two-visible-GPU test confirms
physical output/gradient placement on the nonambient input GPU using the CUDA
pointer attribute, rather than trusting only device labels.

New candidate full/resume traces reuse the fresh same-card genuine Torch
references above. The original producer, inputs, comparison budgets and
checkpoint binding remain unchanged. All four new candidate runs complete
with zero fallback and the required actual update inventories.

| Profile after both arithmetic repairs | Next-token logits max abs | Final adapter error L2 / Torch update L2 | Jittor exact resume | Resume logits max abs |
| --- | ---: | ---: | --- | ---: |
| FP16 | 0.215087891 | 4.333582336% | Fail | 0.19140625 |
| BF16 | 2.375 | 13.277690558% | Fail | 2.19921875 |

Strict parity still fails: 3,175 / 3,358 floating observation fields fail in
FP16 / BF16, separately from 1,120 live-counter representation mismatches in
each full comparison. Own-Jittor continuation fails 2,251 fields per profile;
own-Torch continuation remains exact. Top-1 agrees on all three evaluation
prompts in both precisions. The smaller adapter-update error does not mean all
metrics improve: BF16 logits and both own-resume maxima are worse than before
the scalar repair. This fixes a proven primitive discrepancy but is not a
complete explanation or acceptance of full-model numerical differences.

The new receipts are `scalarfix-practical-run-summary.json`,
`scalarfix-{fp16,bf16}-{cross,jittor-resume}-comparison.json`,
`scalarfix-*-adapter-update-error.json`, `scalarfix-*-final-logits-diagnostic.json`
and `jittor-cuda-*-scalarfix-*.launch.json`/`.source.patch`. Adapter-update
metrics use all 112 tensors after **step 10**, relative to the shared initial
adapter archive SHA256
`25a0f1f55be18f546ef6a1da41def3a405c1f2224e0a348a65fd7083af267624`.
The earlier step-9 diagnostic files are not final-update evidence.

Continuation inspection finds a difference already in the first resumed
forward loss at step 6; this precedes its AdamW update. On-disk checkpoint-5
adapter and optimizer arrays match their recorded step-5 values. Inputs,
RNG-state records, scheduler and scaler state pass the existing restore guards.
Fresh uninterrupted repeats of both current AMP profiles are byte-for-byte
identical over all 5,649 retained fields, with zero fallback and ten actual
updates. See `scalarfix-{fp16,bf16}-jittor-repeat-comparison.json`. This differs
from the older FP32 repeat's small drift and makes a blanket appeal to
nondeterminism insufficient for these two AMP continuation failures. A single
paired repeat does not establish universal determinism.

The accepted producer does not snapshot loaded state before the resumed
forward, so a separate FP16 diagnostic adds a read-only callback after official
model/optimizer/scaler loading, at global step 5. It checks the actual model
parameters and optimizer parameter references, then intentionally stops before
any forward or optimizer update. All 112 actual LoRA parameters, 224 Adam
moments and 112 serialized FP32 counters match checkpoint 5 byte-for-byte:
**448/448 fields**, zero fallback, with matching model/optimizer parameter
objects. The adapter tensors are physically on CUDA. Loaded Adam moments are
on CPU before the native pre-step placement; this alone is not a placement
failure and cannot explain the already observed pre-update loss difference.

This diagnostic rules out incorrect saved/loaded values for the measured FP16
state, but does not establish a BF16 post-load result or a complete training
pair. The observer adds synchronization and changes execution history, so it
cannot stand in for the unmodified producer's continuation result. Public live
counters still expose the separate Python-float API gap.
The raw snapshot's comparisons against the live trace show 112 dtype-only
counter mismatches; the checkpoint-byte audit compares the actual FP32 saved
counter representation and passes all 448 fields. The evidence is
`fp16-postload-snapshot-observations.json`,
`fp16-postload-checkpoint-binding-audit.json` and their wrapper/producer hashes;
`resume-first-difference-audit.json` contains the earlier read-only inspection.
Remaining investigation concerns the restored execution history and precision
path from the first resumed loss onward; no root cause is claimed yet.

## Remaining native BF16 and optimizer boundaries

On SM75, the selected native cuDNN BF16 convolution route fails for the recorded small convolution, while the independent genuine Torch BF16 control succeeds on the same GPU. This is a route-specific unresolved failure, not blanket hardware BF16 unsupported. See `amp2-torch-convolution-capability.json`, `amp2-native-bf16-conv-baseline-r2.log` and the FP32 control.

The explicitly selected generic BF16 grouped 3D convolution route also has a preexisting weight-gradient difference: native `dw=794624`, genuine Torch `dw=798720`, reproduced on CPU/CUDA with `amp_reg=0` and the old/current contraction scopes. This does not diagnose the cuDNN grouped route. The dyadic-input guard fixture isolates input casting without relaxing tolerance or claiming this native discrepancy fixed. Exact shapes, controls and exclusions are in `amp2-final-summary.json` and `amp2-native-grouped-bf16-*-baseline-observations.json`.

The Adam live-counter finding is also still open. Independent live-state probes distinguish Torch's stable mutable scalar tensor from Jittor's freshly returned Python float. NumPy converted that float to FP64 in the earlier traces; that was not evidence that the saved counters were FP64. `adam-counter-existing-checkpoint-audit.json` verifies all 112 checkpoint-5 counters in each of the four original profiles were already scalar FP32 and equal to five. Counter value canonicalization does not establish live identity, aliasing or in-place mutation compatibility.

## Reproduction

Use the isolated candidate/oracle environments and offline artifacts described in
[the original protocol](2026-10-23-trl-full-and-practical.md). Set separate
`JITTOR_HOME` directories per process/device and run native and Torch-mode tests
in different processes. Representative maintained commands are:

```bash
# Native mode; use the intended CPU/CUDA environment and fallback guard.
<jittor-python> -m pytest -q tests/optim/test_sgd_lazy_momentum.py
# Torch mode; select the actual device with JITTOR_TEST_DEVICES.
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  python/jittor/compat/tests/torch/test_torch_scalar_multiply.py \
  -k scalar_multiply_opmath_and_gradient
# Recover into a separate, initially absent destination; sources stay intact.
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_trace_archive.py \
  <archived-trace> --restore-to <new-raw-trace>
```

For the full SGD recheck, use `trl_full_probe.py --help` with
`--algorithm online_dpo` or `--algorithm rloo`, `--device cuda`, the fixed initial
artifacts and restored oracle trace; the exact accepted argument vectors are
in the `*-cuda-sgdfix-10-r2.launch.json` receipts. For LoRA use
`trl_practical_probe.py run` in the independent interpreters, then
`trl_practical_compare.py --reference <torch-trace> --candidate <jittor-trace>`.
Resume a fresh process from each runtime's `trainer/checkpoint-5`; compare it
to its own uninterrupted trace with `--resume`. Source/tool hashes, checkpoint
hash binding and full metric inventories are required; importing successfully
or observing a single matching output is insufficient.

## Durable references and bounded cleanup

`durable-oracles/online_dpo-torch-cuda-10-r2/` and `durable-oracles/rloo-torch-cuda-10-r2/` contain restored references totaling approximately 9.50 GiB (10,202,695,866 bytes). Each `restore-complete.json` binds the original manifest and recovery journal, verifies reconstructed/copied data, and records `source_unchanged=true`. Source archives were not rewritten or removed by restoration. This is durability for these two references, not a persistent backup of the entire earlier matrix.

Separately authorized cleanup of twelve superseded runs removed 6,196 files occupying 67.169 GiB; the observed filesystem free-space increase was approximately 67.154 GiB. It verified 42,508 retained files. The exact plan, deletion journal and completion receipt remain under `$JITTOR_LAB_ROOT/_state/trl-cleanup-20261024/`. This cleanup must not be confused with lossless restoration: the selected obsolete payloads were deleted, while the retained evidence and protected references were checked explicitly.
