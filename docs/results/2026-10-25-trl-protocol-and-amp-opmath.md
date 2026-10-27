# TRL maintained protocols and AMP loss opmath

- Status: measured, bounded text-Trainer admission; whole-library and mixed-precision numerical equivalence are not claimed.
- Date: 2026-10-25–26 UTC.
- Baseline: `e17ddf2b666ecf6127a331c72352174edd5d5b11`, on `2.0-refactor` at `ca56aa802c2acf9a22bca32d1bee3276381ed203`. Measurements bind later working-tree sources separately.
- Owner: native loss, Torch compatibility and downstream gate maintainers.
- Review when: TRL/dependency pins, AMP, native loss, optimizer, producer or CI contracts change.

This report extends the [full-model and repeatability results](2026-10-25-trl-ppo-and-loss-determinism.md).
All raw artifacts below are **unversioned**, relative to `$JITTOR_LAB_ROOT/_state/`.
Pins remain TRL 0.21.0 at `46d09bd2408f17605409fb3ee8ba12705add7faa`, Transformers
4.56.2, Accelerate 1.10.1, PEFT 0.17.1, datasets 3.6.0 and NumPy 1.26.4.

## A real AMP cross-entropy defect

Replaying actual first-batch logits `[1,7,151936]` isolates the loss boundary.
The official Trainer counts seven nonignored labels before shifting, while
causal shifting leaves six predictions. The replay therefore uses the official
sum divided by the CUDA int64 count seven, with FP16 loss scale 1024 or BF16
scale one. Genuine Torch 2.8.0 reproduces the captured scalar loss and all
1,063,552 logits cotangents bitwise. A default-mean replay dividing by six is
retained as a rejected calibration attempt, not a defect.

The native `_CrossEntropyRows` custom Function restores its captured AMP
register when backward builds its graph. That register can narrow arithmetic
on the saved FP32 logits/statistics even after the outer autocast region exits.
The CPU composite forward can also narrow intermediate statistics. The fix
protects this native loss's forward statistics and closed-form backward with
`amp_reg=jt.amp_flags.disabled` (value zero), retaining the existing formula, saved tensors and final input-dtype
cast. It does not change a Trainer or the global native AMP policy.

On the identical real inputs, the backward-only counterfactual reduces FP16
cotangent relative L2 error from `1.48322e-3` to `9.12595e-8`, and BF16 from
`1.25143e-2` to `2.33874e-11`. These are local derivative results, not ten-step
training acceptance. The FP16 counterfactual still has one coordinate outside
the original elementwise budget and 94 unequal elements; BF16 passes that
budget with 18 unequal elements. Neither is a bitwise-parity claim.
CPU and CUDA regressions reproduce the defect before the
fix and pass after it, covering FP32/narrow inputs, scaled cotangents, ignored
targets, retained backward, dtype, residency and zero fallback. Native closed-
form/repeatability regressions pass CPU eight cases plus one CUDA skip and CUDA
nine cases. CPU AMP-boundary tests pass all 19 cases. An expanded CUDA run hits
a separate SM75 BF16 cuDNN convolution failure (KI-BACKEND-021); it is not a
passing full-file run.

Evidence: `trl-backward-20261025/amp-ce-opmath-red-counterfactual-proof.json`,
SHA256 `316ba3db066ec443a2f0aa3f3224dc53372f6767fbbea4037540858039206943`.
It binds exact producers, original/counterfactual arrays, and red/green logs.

## Shared frontend autocast policies

Independent genuine-Torch CPU/CUDA sweeps each execute 250 small API cases.
The unrepaired frontend has 38 CPU and 68 CUDA output-dtype violations. A shared
compatibility helper now protects the measured `log`, `rsqrt`, `exp`, `sum`,
`cumsum`, `cumprod`, softmax and log-softmax entry points. Tensor, functional
and module spellings use the same owners. CUDA's eligible narrow inputs are
widened before computation; explicit dtype/out selects computation dtype first;
in-place calls preserve their target dtype and reject unsafe integral targets.
FP32/FP64 exp backward and explicit FP32 softmax backward also retain wide
opmath. The native CPU narrow-exp output rule is handled at the public frontend boundary below.

The sweeps, final 24-case exp/in-place controls per device, and source/AST
bindings retain their actual producer identities. CUDA's 240 measured default-
policy numerical fields pass the original budgets. Explicit low-precision
requests still have separate strict forward/VJP differences; the largest
`0.01682452` VJP discrepancy is a BF16-output/FP64-input cumprod request, not
the default CUDA FP32-policy path. No blanket half-precision numerical
equivalence follows. Twelve integral in-place calls reject without changing
bytes, and eight FP32/complex64 calls retain oracle behavior.

CPU's complete AMP-boundary file passes 24 tests; CUDA passes the other 23 when
the separately reproduced KI-BACKEND-021 cuDNN case is explicitly deselected.
The earlier full-file failure is preserved. Evidence is
`trl-backward-20261025/amp-public-policy-final-audit.json`, SHA256
`f6690b7e2326c930f37e74afc23cc2d5d0761a59050a395218a1b9dd8d0f6b95`.

## CPU exp, in-place writes and returned VJP dtype

The native CPU exp intentionally computes narrow inputs in FP32. Its Torch
frontend now narrows the public result to the input dtype, including when the
active CPU autocast dtype differs. This preserves the native opmath. The
existing in-place unary owner also handles exp_: target dtype, object identity
and materialized aliases are preserved, while integral targets reject before
mutation. Unsupported exp out= calls still raise their original errors.

The public autograd.grad owner now returns each nonmissing gradient in the
corresponding input dtype, as Tensor.backward already does. Wider internal
arithmetic is retained. The cast stays differentiable for create_graph=True;
unused None entries and the existing materialization policy are preserved.
Regressions include mixed input dtypes, unused/materialized zeros and reachable
second derivatives. They do not establish all materialization semantics.

The 63-case independent Torch control finds 24 output-dtype and twelve
returned-gradient-dtype violations before repair. Afterwards, all 36 supported
calls and thirty VJPs have the expected dtype, and their 66 numerical fields are
bitwise equal for these fixed inputs. Six invalid dtype= calls still reject;
eighteen unsupported out= calls retain their original exceptions; three integer
exp_ targets now reject without mutation.

The affected CPU suite initially passes 80 tests with one new fixture failure;
CUDA passes 79 with that same failure and one explicitly deselected
KI-BACKEND-021 case. The fixture compared native and frontend dtype string
spellings; normalizing the spelling and removing an incorrect assumption about
unchanged materialization policy makes that sole node pass on CPU and CUDA.
Production sources are identical across these runs. The original failed runs
remain retained. Evidence:
`trl-backward-20261025/exp-autograd-final-audit.json`, SHA256
`209c1bde7b21f10b25b9bb2a55098802f50e837082382ffcc4869a3ee21d8f3e`;
`exp-cpu-contract-final-audit.json`, SHA256
`2ee215bb6854285f955986b647e17d5e08d0567a67c54671bae8d74bff3dd604`.

## Complete online loss observations

Tiny PPO/RLOO originally retained full gradient/parameter trajectories without
direct scalar-loss observations. A negative test removes the loss on both
sides and demonstrates that the old collector accepts the incomplete pair.
The producer now observes the unchanged official `accelerator.backward` loss,
forwards its arguments/result/exceptions and restores the method on exit. Both
the producer and independent collector require loss at every actual update.

New independent CPU pairs execute three updates per runtime, with zero fallback:

| Tiny FP32 method | Compared fields | Complete gradient/parameter fields | Scalar losses | Loss max abs | Strict result |
| --- | ---: | ---: | ---: | ---: | --- |
| PPO | 375 | 147 / 147 | 3 | `1.25030056e-7` | Pass |
| RLOO | 219 | 72 / 72 | 3 | `9.31322575e-9` | Pass |

Both methods have nonzero aggregate gradients and parameter changes at all three
updates. RLOO's first two scalar losses really are zero in both runtimes; an
undefined zero-vector cosine remains null. The 85 related pure tests and Ruff
pass. These pairs precede the later loss repair and retain that source boundary.
Evidence: `trl-pr-admission/loss-observer/completion.json` and its bound receipts.

## Source-frozen text matrix and complete-model continuation

On the final frozen runtime sources, after all repairs in this report, the
actual canonical nox entry passes all **16 / 18 @ upstream `46d09bd2408f17605409fb3ee8ba12705add7faa`**
selected text-method pairs independently on CPU and CUDA. Each method performs
three actual calls per runtime: 48 calls per runtime/device, 1,236 full-gradient
fields, 1,236 updated-parameter fields and 48 directly observed scalar losses.
CPU compares 3,700 fields, CUDA 3,697; all pass the original budgets with zero
fallback. Both runs bind the same 1,277-file runtime and 33-file producer maps.
These remain tiny FP32 Qwen3 configurations, not sixteen pretrained methods or
a generic scorecard L-level award. Raw evidence is under
`trl-pr-admission/final-source-{cpu,cuda}16/`, including numerical summaries, complete
payloads, actual environments, nox logs, source maps and command receipts.
`final-source-cpu-cuda-comparison.json` verifies both source maps against the
current runtime. The 1,277-file runtime-map SHA256 is
`369948a2f71aca7db3ed9793bc992d226d20d38508a3ba0c887176f242bec5f5`;
the 33-file producer map is unchanged. CPU
`final-source-cpu16-canonical-audit.json` binds the fresh actual summary and
all 32 worker manifests.

The unchanged full Qwen3-0.6B SFT producer also completes both precision profiles:
ten actual LoRA/AdamW updates and a fresh-process checkpoint-five continuation
through updates six to ten. Each precision's own continuation is bitwise equal
in all 2,829 checked fields, with zero fallback. Reused genuine-Torch references
retain their original source identities. Full cross-runtime strict comparison
still fails, with no relaxed tolerance:

| Full CUDA SFT | Failed cross-runtime fields | Adapter error / actual Torch update | Next-logit max abs | Maximum prompt KL | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 3,136 | 3.737774% | `0.265625` | `0.00082280794` | 3/3 |
| BF16 | 3,338 | 8.396626% | `1.59375` | `0.01718547124` | 3/3 |

The KL/logit results describe three fixed final prompts, not model quality.
The remaining BF16 distribution difference is material. Repairing verified
local opmath defects does not establish full training equivalence or show that
every remaining discrepancy is harmless rounding. KI-COMPAT-018 stays open.
Before the first optimizer update, complete 112-adapter gradient relative L2
error is already 0.783352% in FP16 and 1.453669% in BF16; at update ten it is
1.161215% and 8.222113%. This rules out attributing all differences solely to
optimizer state. `trl-pr-admission/final-amp-global-gradient-diagnostics.json`
binds the checked complete tensors at updates one, five and ten.
At the first update all 56 A gradients are zero in both runtimes; the differences
are in the 56 B gradients. The v projections contribute about 63.5%/61.2% of
FP16/BF16 error energy. Same-input actual-CUDA A-forward replays across all 28
layers reproduce 224 fields per runtime byte for byte, with zero fallback.
This excludes that local replay as a cause, but does not establish identical
actual model inputs or upstream cotangents. Evidence:
`trl-backward-20261025/ceamp3-lora-A-and-gradient-boundary-final-audit.json`.

An actual candidate FP16 observation then binds the same 423 initial tensors,
batch, scale 1024 and every one of the 112 formal first-gradient byte strings.
It stops before AdamW: one backward, zero actual updates. At layer0-v the real
RMS output `[1,7,1024]` and B input `[1,7,8]` match Torch bitwise, while B's
incoming scaled cotangent differs in 6,015 / 7,168 elements (relative L2
1.066872%). Its B-weight gradient differs by 0.998353%, contributing 12.0364%
of complete first-gradient error energy. Both runtimes reproduce both formal
B gradients bitwise using the corresponding actual cotangents and original
GradScaler.unscale_. Exchanging cotangents exchanges the resulting gradient;
exchanging these identical inputs does not.

This locates this site's discrepancy upstream of B's local weight VJP and the
optimizer. It does not identify the first divergent operator: later-layer
forward differences, backward arithmetic, the PEFT scale chain and the loss
seed remain possible contributors. It is not a normal-rounding conclusion.
The observation and cross-replays retain 269 hash-checked arrays in about
10.2 MiB, without modifying runtime or formal producers. Evidence:
`trl-backward-20261025/ceamp3-layer0-v-actual-cotangent-final-audit.json`, SHA256
`f6db8e74ea65cf461a3b953afd598d3751ad5b534a5085bcf505eea911b9d3ba`.

Evidence: `trl-local-fixes/ceamp3-practical-source-binding.json`, four
`jittor-cuda-{fp16,bf16}-ceamp3-{10,resume}/trace/` runs, and
`ceamp3-{fp16,bf16}-{cross-comparison,jittor-resume-comparison,adapter-update-error,final-logits-diagnostic}.json`.
The final full runs also reproduce all 5,649 earlier ceamp2 arrays per precision
byte for byte, including signed zero, after validating their manifests and
actual bytes (`ceamp3-vs-ceamp2-candidate-bytes-audit-r2.json`). The complete
gradient diagnostic values above therefore remain unchanged on final sources.

The native CPU narrow `exp` output rule is subsequently corrected at the
frontend boundary (see below); this does not imply complete CPU low-precision
parity. The original intermittent CUDA generation failure also remains open,
even though this complete source-frozen CUDA gate passes.

## Canonical admission and diagnostic entry points

`nox -s ecosystem_trl` runs the existing complete sixteen-text-Trainer,
three-update protocol. Only fresh output and exclusive cache paths can override
the wrapper; method/step overrides fail before execution. The CPU nightly uses
the same underlying gate. Unified `registry_inventory.py --repo trl` reuses the
source-only AST extractor, including optional diffusion exports: the full
denominator is **18 @ upstream `46d09bd2408f17605409fb3ee8ba12705add7faa`**.
The unified audit discovers the actual dedicated CLI and nox session, with
`verified_level=null`; registration does not fabricate a generic scorecard
numerator or pytest node. See the [runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/trl-torch-compat/SKILL.md).

The maintained `trl_generation_index_probe.py` offers public-mask and unchanged-
producer PPO observations. Metadata mode changes timing; values mode explicitly
synchronizes and checks the small indices/target/output. Its latest CPU/CUDA
mask pairs pass. A fresh current-producer CUDA PPO pair passes all 375 fields,
three actual updates and zero fallback; all 41 observed gather values, outputs
and native input-ID bindings are correct. Twenty-eight pure guard tests and
Ruff pass. Evidence: `trl-ddp-local-20261025/maintained-generation-index-final-r4/diagnostic-entry-receipt.json`.
This provides a reproducible diagnostic entry, **not a repair or closure of
KI-COMPAT-021**. The original intermittent failed batch remains retained.

## Low-precision CUDA softmax backward

A separate saved-output/cotangent replay isolates premature rounding in the
register kernels. Half/BF16 softmax retains the same low-precision temporary
product as Torch, then evaluates the final subtraction in FP32 before one
output cast. Log-softmax similarly evaluates its exponential and final
subtraction in FP32. This is conditional on the narrow dtype; it does not
change FP32 kernel generation or introduce a second frontend implementation.

The maintained regression fails before the repair: four FP16 gradients become
zero where the independent reference is nonzero. After repair, all six native
CUDA tests pass with no skips. Eight genuine-Torch same-saved-output/cotangent
comparisons are bitwise equal in all 32 elements, with zero fallback. Pure
source comparisons bind 54 unchanged forward and 18 unchanged FP32-backward
kernel strings; streaming and cross-entropy function text/AST are unchanged.
This is kernel-local evidence and does not replace the complete-model AMP
results above. Evidence:
`trl-backward-20261025/softmax-register-product-final-audit.json`, SHA256
`f230277532e29b93fe3816e606a399d9ab2638b169536080bf7816ec122c29c2`.

## Repository admission and test isolation

The first complete CPU smoke invocation finishes with 8,862 passes, 60 failures,
83 errors, 3,782 skips, 11 xfails and one xpass; its exit status is one. It is
**not a passing repository gate**. Its command is
`python tools/run_test_suite.py --tier smoke --backend cpu`, using the default
four workers and parallel compiler with separate native/Torch processes. The
retained outer environment additionally disables MKL and forces fallback errors;
those settings affect negative unit tests and are not intrinsic to this CLI.
The cold invocation takes 6,824 seconds and retains its original source map,
logs and environment under `trl-pr-admission/smoke-canonical*`.

Definition-only compatibility imports do not establish frontend activation.
Native fixtures now verify actual publication/install state, the canonical Var
identity and native graph policy. Positive independent children allow inactive
definitions and genuine Torch; a real activated-shim child is rejected even
when Var identity remains native. TP checks only fallback events during its own
protocol and retains both absolute counters. New-event negative controls still
fail. The combined affected native CPU selection passes 182 tests, zero skips;
no runtime arithmetic or numerical budget changes for these test repairs.
Evidence: `trl-pr-admission/native-runtime-guards-final-receipt.json`.

The runner accepts an explicit absolute `JITTOR_TEST_TMP_ROOT` outside every
Git checkout while retaining its existing separate JIT caches. This supports a
lab nested under an unrelated ancestor checkout without weakening artifact
ownership guards. Seven runner-contract tests pass with an external temporary
root. A named zero AMP policy also removes seven bare-register style failures;
its regression preserves explicit FP64 arithmetic under an outer AMP scope and
restores that scope. The AMP-register and isolated post-fork compiler selection
passes 18 tests. The isolated fork result does not explain the earlier smoke
child's eight-second timeout.

The final standalone structure gate passes **1,528 tests, four explicit skips**
after the identical native fixture bodies are extracted into their shared scope;
no duplicate-code waiver is added. The corrected, source-frozen native selection
passes all **182 tests, zero skips**. The six tool files pass **208 tests, zero
skips**, in their canonical native process mode; the Conv2d marker case passes
separately in Torch mode. Four structural skips cover two backend-resolver-only
names and two unavailable-CUDA structural branches. The accepted measurements
retain 2,588 common identical source entries; no runtime or tests change between
them. Evidence: `trl-pr-admission/corrected-final-baseline-admission-audit.json`.
A native run missing the required independent-oracle variable skips five cases;
that narrower result is retained and is not substituted for the complete run.

## Independent upstream failure controls

A clean detached worktree at the report's upstream baseline runs the exact
same twenty originally failing Torch nodes in a separate CPU build/cache,
interpreter and oracle configuration. Nineteen failures reproduce, including
the existing factory/leaf, native-checkpoint copy, view/lifetime, replay and
missing-VERL-environment errors. The Conv2d bias case is an unexpected success
in both trees: its dtype assertion passes and its stale expectedFailure marker
causes failure. The marker and obsolete open-issue entry are removed after
independent verification; no new Conv2d arithmetic repair is attributed here.

The one upstream pass exposes a genuine branch regression: integral sum now
correctly returns int64, but bucketize(out_int32=True) assumes its internal sum
is already int32. Bucketize now explicitly casts its result to the requested
width, preserving the corrected public sum behavior. The retained CPU red has
two failures and one passing integer-sum control. CPU and real CUDA device
controls then each compare sixteen dtype/side/output-width combinations with
genuine Torch, plus two integer sums including 2**30 + 2**30 = 2**31. All pass,
with physical-residency checks. The old numerical-fidelity CPU node also passes;
its execution in the CUDA launch is still CPU evidence. Evidence:
`trl-pr-admission/bucketize-regression-final-receipt.json`.

The clean upstream native mixed-precision file separately has five passes and
one failure. Its no-MKL full-BF16 MLP prints BF16 loss `0.22852` and FP32
`0.12238` (five decimal places), with the same full reported excess ratio
`0.8672143515` as the original smoke. This is not an
autocast/FP32-master profile, and its cause remains unlocalized (KI-OPS-017).
The original 25% budget is unchanged. Test sources are byte-identical across
all fourteen compared files. Both independent builds and the tests retain clean
source snapshots. Total additional worktree/artifacts are about 172 MiB, with
no existing caches copied or removed. Evidence:
`trl-pr-admission/corrected-final-baseline-admission-audit.json` and its bound
receipts/XML/logs, specifically `final-r6-origin-native-mlp` for the native
control. Earlier launches carrying explicit frontend project/runtime roots
activate the shim even with its switch disabled; those native launches are
invalid and remain retained as such. The accepted native controls remove those
roots and verify native policy, publication and install state before and after
execution. A baseline failure remains a failure; these controls establish
attribution rather than a passing whole-repository result.

## Evidence storage

Four historical DPO/GRPO CPU/CUDA oracle traces retain all 1,240 bulk arrays as
verified lossless Zstandard streams, with original byte hashes and recovery
journals. Only the verified redundant uncompressed copies are removed, freeing
2,466,763,984 bytes. `trl_trace_archive.py --restore-to <fresh-external-path>`
recovers the original trace bytes for consumers requiring plain NPY files.
Evidence: `trl-pr-admission/lossless-space/receipt.json`. Weights, binary caches
and raw arrays remain outside the PR.
