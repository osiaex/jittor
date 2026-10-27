# Full-model PPO LoRA and CUDA loss repeatability

- Reviewed: 2026-10-25.
- Owner: Torch compatibility, native loss and CUDA backend maintainers.
- Recheck when: PPO/PEFT/library pins, model roles, optimizer or loss arithmetic, reduction scheduling, producer sources or measured hardware changes.
- Status: matched full-model PPO LoRA CPU/CUDA runs complete; strict numerical parity fails. CUDA row-LSE and the measured full-model DDP repeat/resume nondeterminism are repaired. Strict cross-runtime parity and an intermittent tiny-PPO CUDA generation failure remain open.
- Source baseline: `527cb3205ae4701447984ed2c5b639f15b65d9f9`, synchronized with Jittor `2.0-refactor` at `ca56aa802c2acf9a22bca32d1bee3276381ed203`. Each measurement binds its actual source hashes; the commit alone does not identify later working-tree changes.

The [earlier text/LoRA report](2026-10-25-trl-text-and-lora.md) contains the historical, source-bound 32-cell tiny-model matrix and separate SFT throughput measurements. Those results are not reruns of the loss repair below. The PPO pairs in this report also completed before that repair and retain their original runtime fingerprints.

Artifacts below are **unversioned**, outside Git, relative to `$JITTOR_LAB_ROOT/_state/`. Pins are TRL 0.21.0, Transformers 4.56.2, Accelerate 1.10.1, PEFT 0.17.1 and NumPy 1.26.4. PPO oracles use genuine Torch 2.8.0+cu128; installed TRL is verified against upstream `46d09bd2408f17605409fb3ee8ba12705add7faa`. Cross-runtime comparisons retain `atol=1e-5, rtol=1e-4` and the existing field-scale floor. Controlled discrete inputs are compared exactly. No tolerance is relaxed.

## A separate PPO configuration fits the available CUDA memory

Official `PPOTrainer` supports a PEFT policy with `ref_model=None`: its reference context disables policy adapters, avoiding a fourth complete model. The new probe uses three complete pretrained, 28-layer Qwen3-0.6B FP32 bases: policy, value and frozen reward. The policy uses `get_peft_model`; the value model uses PEFT's `inject_adapter_in_model`, preserving the native classifier/backbone contract, with its score head explicitly trainable. Reward and initial value heads are shared diagnostic fixtures, not calibrated reward models.

Rank-8/alpha-16 LoRA targets `q_proj` and `v_proj` in both policy and value, with zero dropout. Training covers **224 adapter tensors plus one value-head tensor: 225 tensors and 2,294,784 scalar parameters**. All other weights remain frozen and are hash-checked. The optimizer is AdamW, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, constant scheduler and no gradient clipping; attention is eager.

The CPU prerequisite executes one actual update from a two-prompt rollout. CUDA executes five two-prompt rollouts with two minibatches and one PPO epoch each: ten actual optimizer calls visiting all ten training prompts. Native generation and floating-point calculations remain active; only sampled discrete token choices replay the independent oracle. Initial adapters, heads, buffers and all model-role hashes match across each pair. The CPU and CUDA oracle initial role bytes also match.

Both runtimes complete the CUDA profile on an 11 GiB SM75 card, with zero fallback. This establishes feasibility of this **three-base LoRA profile**. It does not close the original four-role, full-parameter PPO CUDA capacity failure. No peak-memory ratio, timing result or checkpoint-resume coverage is claimed here.

## Complete trained-tensor comparisons

Every actual optimizer call retains all 225 gradients, updated parameters, Adam counters and both moment tensors in both runtimes. Scalar losses are observed at the unchanged official `accelerator.backward` call. The independent audit derives the trainable inventory from the initial policy/value roles, checks payload hashes and shapes, recomputes every comparison, and verifies generation replay and unchanged runtime fingerprints.

| Profile | Actual calls, each runtime | Failed / observed fields | Final full-gradient max abs | Final trainable-update relative error | Next-logit max abs | Mean softmax KL, Torch → Jittor | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU FP32 prerequisite | 1 | 23 / 1,385 | `6.43730e-6` | 0.151609% | `2.16961e-4` | `2.74301e-10` | 10/10 |
| CUDA FP32 LoRA PPO | 10 | 238 / 13,625 | `4.62532e-5` | 0.0868595% | `1.79291e-4` | `2.58239e-10` | 10/10 |

Trainable-update relative error is `L2(Jittor final − Torch final) / L2(Torch final − shared initial)`, over all 225 trained tensors, including the value head. Top-1 and KL refer to ten fixed final next-token distributions, not training quality.

CPU's 225 full gradient tensors and both moment groups pass the original budget; eight updated parameter elements in eight tensors fail. CUDA retains 2,250 full gradient tensors: 40 fields fail over the complete trajectory. At the final step, one gradient field/two elements fail; its global relative L2 error is `1.91146e-5`. Final parameters fail in 12 fields/46 elements. All ten CUDA scalar-loss fields, all 2,250 counter fields and both 2,250-field Adam moment groups pass. Every step has nonzero aggregate gradients and a nonzero parameter update in both runtimes. Both complete runs nevertheless **fail strict overall numerical comparison**, including all ten final-logit fields.

The one-minibatch CPU control logs `val/ratio_var=NaN`: upstream PPO computes a sample variance from one statistic. This is an undefined logging metric, not a nonfinite training loss, gradient or optimizer state. The two-minibatch CUDA profile does not have that particular logging degeneracy.

Evidence is under `trl-ppo-lora-20261025/`: `cpu-1-r2-final-audit.json`, `cuda-10-r2-final-audit.json`, the four `ppo-lora-{torch,jittor}-{cpu-1,cuda-10}-r2/trace/` directories, and archived exact producers/tests in `measured-v2/`. The earlier CUDA capacity oracle remains separately labeled `ppo-lora-torch-cuda-10-r1`, with `measured-capacity-v1/`. All five retained runs total 1,451,958,315 bytes; `completion-storage-receipt.json` binds the final audit receipts. Maintained tools are `trl_ppo_lora_probe.py` and `trl_ppo_lora_case.py`; their 12 pure inventory tests and Ruff pass. Reproduction commands are in the [runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md#full-model-ppo-with-policyvalue-lora).

### What the eight CPU parameter mismatches establish

All eight offending first-update coordinates start at zero in LoRA B. Applying the ideal first Adam update `-lr*g/(abs(g)+epsilon)` independently to each runtime's recorded gradient explains the eight observed parameter differences with maximum residual `6.94e-12`. At the largest difference, Torch's gradient is `-4.22445e-8` and Jittor's is `+3.99960e-8`; the parameter difference becomes `1.608576e-4`, with formula residual `2.32e-12`.

`cpu-first-adam-coordinate-diagnostic.json` records all eight coordinates. This establishes downstream amplification at those coordinates. It does not establish the origin of every gradient difference, classify the complete trajectory as harmless rounding, or change strict admission.

## CUDA cross-entropy: exact loss can hide a changing backward

A separate native loss reproducer fixes FP32 logits of shape `[16, 151936]` and labels, then performs 32 CUDA forward/backward repetitions without an optimizer or collective. Jittor's scalar loss is identical in every repeat, but **26 of 31 comparisons with the first repetition** have different saved per-row log-sum-exp values and gradients. The saved normalizer changes by up to `1.9073486328125e-6`; gradient changes reach `1.1175870895385742e-8`, affecting up to 443,598 elements in one comparison. Genuine Torch's loss and gradient remain exact across all 32 repetitions.

The native `_CrossEntropyRows` backward uses the saved row normalizer. The executed-kernel profile identifies the fused shifted-exp sum for `[16,151936]` and retains its generated source containing `atomicAdd`. Partial sums can arrive in a different order; the rounded scalar loss can remain identical while saved normalization values and their downstream gradients differ. This establishes a local nondeterministic loss component without requiring distributed communication. It does not, by itself, prove that this component explains every DDP continuation discrepancy.

Red evidence: `trl-ddp-local-20261025/ce-{jittor,torch}-cuda-r1/manifest.json`, `ce-kernel-source-r1/executed-source-audit.json` and the retained generated source. The exp-sum source SHA256 is `0e8ebe876bc4e4d9b1129a51d3171d206b053b41bb54dfbdf05977f6f30b9085`; the separate scalar reductions are not confused with that vocabulary reduction.

The implemented repair adds an optional CUDA row-LSE operation in `backends/cuda/kernels/nn/softmax_cuda.py`, selected from native `python/jittor/nn/functional/loss.py`. One thread block owns each row and writes only `[N]`, sharing the existing streamed max/exp-sum reduction body. It avoids a vocabulary-sized intermediate. Admission is limited to contiguous, nonempty, two-dimensional FP32 CUDA inputs with at most `INT_MAX` elements; low-precision CE inputs already widen to FP32, while CPU, FP64 and unsupported layouts retain the prior composite. The saved tuple, closed-form backward, target weighting and public reduction wrapper remain unchanged.

Independent source review confirms byte-identical generated code for existing streamed softmax modes and unchanged ASTs for its backward, scheduling and register path. `softmax-factor-proof.json` records the factoring proof. Nonfinite rows retain the old propagation rules; this is not a repair of ignored-row/nonfinite semantics. The intended repeatability claim concerns the saved row normalizer and measured shapes, not every possible outer loss reduction.

Post-fix CE repetitions now have byte-identical saved LSE, loss and complete gradients in all 32 runs. All 2,430,976 gradient elements pass the unchanged genuine-Torch budget; maximum absolute gradient error is `5.58793545e-9`, relative L2 `4.34548e-8`. Evidence is `trl-ddp-local-20261025/ce-row-lse-proof.json` and the post-fix trace. Existing six native closed-form loss tests pass on CPU/CUDA, covering weights, reductions, ignored targets, spatial/half inputs and retained graphs. New repeatability tests pass two CPU cases with the CUDA-specific case skipped in CPU mode, and all three cases on CUDA. Earlier failures from FP64 auto-conversion and transpose/storage-view fixture assumptions are preserved separately and are not kernel failures.

### Actual full-model distributed repeat and continuation

The unchanged official two-rank FP32 Qwen3-0.6B LoRA producer reruns ten updates, a fresh checkpoint-5 continuation and a separate independent ten-update repeat. Both ranks perform real NCCL training with zero fallback and unchanged source snapshots. Own-runtime continuation now passes all **2,846 fields exactly**, and the independent repeat passes all **5,686 fields exactly**, both maximum zero. Before the CE repair these exact comparisons failed in 1,350 and 4,274 fields respectively. This closes the measured repeat/continuation nondeterminism, beyond the primitive-only evidence.

Cross-runtime comparison still fails 13 of 5,686 fields: ten observations of one adapter coordinate and three final-logit fields. Maximum next-logit error is `0.00010728836`; final adapter error divided by the actual Torch update is `0.0079395694%`. The first near-zero-gradient coordinate still changes sign across runtimes; the same first-Adam formula predicts each observed update within `7.1e-13`. This explains amplification at that coordinate rather than establishing the upstream cause or waiving strict parity.

Use `trl-ddp-row-lse-20261025/`, plus `trl-ddp-local-20261025/{postfix-ddp-stage-receipt,postfix-ddp-final-proof,postfix-ddp-adapter-update-error}.json`. The six measured rank workers retain 627,804,254 payload bytes. Default FP32 DDP is the scope; no compression, FSDP, AMP DDP or multiple-node admission follows.

## Remaining mixed-precision boundary investigation

Actual first-backward captures in genuine Torch reproduce all 112 retained formal adapter gradients bitwise, before any optimizer update. On those same tensors, both FP16 and BF16 unhooked local replays have all 336 LoRA-B output/input-gradient/weight-gradient fields exact across module and functional routes. Both routes' lm_head output and long-vocabulary input VJP are also exact, including dtype contracts. This rules out the tested local projection/LM-head precision hypotheses.

All FP32-input RMS output/dx fields pass the ordinary budget. Remaining local failures are confined to low-precision-input RMS: 15 FP16 and seven BF16 forward coordinates, and 31 FP16/two BF16 input-gradient coordinates. Each pair differs by one low-precision representable step. Explicit manual graphs reproduce all 110 low-precision-input module paths, their final input gradients and their cast seeds bitwise in each runtime. Aggregate raw FP32 input-gradient error against the FP64 reference is approximately `6.7e-8` to `7.4e-8` in both backends; raw backward cancellation can nevertheless magnify ULP counts near zero. This is not evidence that every training difference is harmless.

An external counterfactual retains the native rsqrt forward but uses Torch's primitive derivative evaluation order, with asserted FP32 backward opmath. It does not improve the remaining VJP differences: FP16 failing input-gradient coordinates change from 31 to 33, BF16 stays at two. No runtime patch is justified by that hypothesis. Only the correctly guarded third diagnostic attempt is admitted; earlier observer-plumbing/opmath attempts remain separate.

After the CE repair, the unchanged practical SFT producer reruns full ten-update and checkpoint-5 continuation pairs. Each precision's own resume is exact in all 2,829 fields, zero fallback. Cross-runtime strict comparison remains failing:

| CUDA SFT LoRA | Final adapter error / actual Torch update | Three-prompt next-logit max abs | Maximum prompt KL | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 4.030722% | 0.1572265625 | 0.0003940962 | 3/3 |
| BF16 | 9.011486% | 1.65625 | 0.02883205 | 3/3 |

The BF16 distribution difference remains material despite matching three argmaxes; no quality-equivalence claim follows. Bound evidence is `trl-backward-20261025/backward-boundary-and-postCE-summary.json`, `rms-raw-FP32-offline-audit.json`, `rms-rsqrt-counterfactual-summary.json`, and `trl-local-fixes/cefix1-{fp16,bf16}-{adapter-update-error,final-logits-diagnostic}.json`. Old pre-CE measurements retain their original source identity.

## Fresh CPU-wheel provisioning boundary

Two new isolated environments have successfully installed the pinned stack with genuine `torch==2.8.0+cpu`; both report no CUDA build, and both dependency checks pass. A fresh native CPU core has warmed successfully. Receipts are `trl-ci-exact-20261025-v1/provision-receipt.json` and `cpu-wheel-identity.json`.

The maintained default three-step CPU gate completes all **16 selected text methods** on the frozen post-CE source: 48 actual optimizer calls per runtime, 3,694 numerical fields, 1,236 gradients and 1,236 updated parameters, zero fallback and all strict comparisons passing. `cpu-exact-final-receipt.json` binds the environment, source, lockfile and summary receipts. Both numerical workers and independent audits use Python 3.11.17; local orchestration uses Python 3.12 and the host compiler/container differ from the remote runner. This validates the pinned CPU installation and numerical entry point locally; no GitHub Actions execution is claimed. Tiny PPO/RLOO scalar training losses are unmeasured, rather than assigned zero; this differs from the new full-model PPO observer above.

A fresh complete CUDA gate then passes all 16 methods on the same frozen native runtime: 48 actual calls per runtime, 3,691 numerical fields, 1,236 gradients and 1,236 parameters, zero fallback. Both final device batches use identical 1,277-file native source maps. `trl-ddp-local-20261025/final-cpu-cuda-gates-receipt.json` binds both completed batches and the earlier failed CUDA batch. The passing batch is `postfix-cuda16-gate-r3`; it does not close the intermittent fault below.

## Intermittent CUDA generation failure remains open

The post-fix CUDA breadth gate completes all 16 genuine-Torch references and its first six candidate methods pass, then the tiny PPO candidate encounters CUDA error 719 during generation. An indexing kernel reports out-of-range values on an axis of length four. This failed batch is preserved at `trl-ddp-local-20261025/postfix-cuda16-gate-r2/`; it is not a passing sixteen-method result.

Exact-environment PPO controls subsequently pass three real updates and the original strict comparison, including five unobserved warm repeats and a separate metadata observer. A GPU1/old-tail-cache control also passes. These successes do not diagnose or clear the original failure. A CUB cumsum participation hypothesis has not produced a red: 100 shared-memory-conditioned repetitions and compute-sanitizer synccheck report correct outputs and zero errors on the tested short/tail sizes. CUB source remains unchanged. A bounded first-use control moves only the 40 artifacts created during the original failed PPO interval into a SHA-bound backup, rebuilds them for one exact-environment run, and restores every original artifact after recording the result. That control passes all three updates and 372 comparisons. Ten generated sources and command keys remain byte-identical; rebuild-artifact differences are not treated as a cause. The public bool-mask/vmap pair also matches genuine Torch across 64 iterations and eight short/padding cases. `ppo-generation-bounded-investigation-receipt.json` binds these results. No global synchronization or cache-clearing workaround is accepted as a fix; the causal fault is still unlocalized and tracked as KI-COMPAT-021.


## Maintained diagnostics and repository checks

The backward-boundary tool now rejects incomplete active-module/cotangent and replay inventories, validates stored array/file hashes, and checks fresh external output paths before writing. Both 794-array captures and all four 225-case/562-array replays revalidate offline; tensor-operation ASTs are unchanged from the archived measured helper. This later guard-only change is bound separately by `trl-backward-20261025/backward-guard-maintained-revalidation.json`; it is not a new numerical run. Its 38 pure tests pass.

The local-loss tool's CPU residency spelling and preceding-step parameter binding are also corrected. A replay must match the selected rank's recorded SHA, shape and dtype. Twenty pure tests pass, and independent CPU CE controls on `[4,257]` repeat exactly three times per runtime; all common fields pass the original cross-runtime budget. Evidence is `trl-ddp-local-20261025/diagnostic-cpu-guard-validation.json`. The original CUDA red/green helper sources remain archived. This does not promote the unmeasured full local-model replay mode to acceptance.

Repository checks: native CPU core 104 passed, 41 skipped, one expected failure; frozen Torch CPU core 243 passed, one skipped; full structure 1,516 passed, four skipped (`trl-local-fixes/cefix-structure-final-r2.log`). A stale generated manifest in the earlier structure attempt was regenerated before that passing run. After the final diagnostic guards, the combined maintained-tool suite passes **199 tests**, and documentation/packaging/collection contracts pass **57 tests**. Logs are `trl-ddp-local-20261025/final-maintained-tool-tests-r2.log` and `trl-local-fixes/cefix-final-guard-contracts.log`. New files pass Ruff; changed owners introduce no new lint diagnostics. Layout and diff whitespace checks pass. Skips are not device coverage, and repository checks do not override any numerical or intermittent-runtime failures above.
