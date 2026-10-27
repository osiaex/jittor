# Local TRL text, LoRA and distributed validation

- Reviewed: 2026-10-25.
- Owner: Torch compatibility and downstream validation maintainers.
- Recheck when: Trainer/library pins, optimizer state or loading, module shortcuts, RMS/AMP arithmetic, producer protocol or measured hardware changes.
- Status: two module precision defects and normal integer Adam counter lifecycle repaired; FP16/BF16 own-runtime continuation is exact. Full cross-runtime strict numerical parity remains open.
- Source baseline: `4bb3e8403e01937e8b64d36a53b62412e37a05ea`, following synchronization with Jittor `2.0-refactor` at `ca56aa802c2acf9a22bca32d1bee3276381ed203`. Measurements bind subsequent working-tree sources through hashes and launch patches; the baseline commit alone does not identify a measured implementation.

This updates the continuation and live-counter conclusions in the [previous local report](2026-10-24-trl-local-fixes.md). It adds bounded text/LoRA configurations, not whole-TRL coverage or L2 admission. The full-model LoRA experiments use complete pretrained, 28-layer Qwen3-0.6B. Their base remains frozen; LoRA trains 1,146,880 parameters in 112 tensors, rank 8/alpha 16, targeting `q_proj` and `v_proj`, with zero adapter dropout.

Pins are TRL 0.21.0, Transformers 4.56.2, Accelerate 1.10.1, PEFT 0.17.1 and NumPy 1.26.4. Independent oracle processes use genuine Torch 2.8.0+cu128. The installed TRL source is verified against upstream `46d09bd2408f17605409fb3ee8ba12705add7faa`. CPU and CUDA comparisons use the same checkpoint, initialization and controlled inputs within each pair. CUDA measurements use the recorded SM75 environment. Correctness rows are small controlled experiments; the separate throughput section defines its own timing protocol. No model-quality benchmark is claimed.

All artifact paths below are relative to `$JITTOR_LAB_ROOT/_state/`. Raw traces, checkpoints, launch patches and JSON receipts are **unversioned**, outside the repository. Cross-runtime comparisons retain `atol=1e-5, rtol=1e-4` and the existing field-scale floor. Controlled discrete inputs and own-runtime continuation use exact comparisons. No tolerance was loosened.

## Final text-Trainer matrix and maintained gate

The pinned upstream public inventory contains 18 Trainer exports. This machine now covers all 16 selected **text** Trainers: SFT, DPO, KTO, ORPO, CPO, Reward, PPO, RLOO, GKD, OnlineDPO, GRPO, PRM, BCO, IterativeSFT, NashMD and XPO. The excluded AlignProp and DDPO exports concern deprecated diffusion training and require a different model/dependency target. “16/18” is an inventory count, not a percentage of every TRL feature or supported version.

A final unified matrix refreshes every candidate against the preserved, hash-checked independent references, on one frozen runtime tree: **16 methods × CPU/CUDA = 32 strict passes**, using the original comparison budget. Each runtime executes 320 actual optimizer calls with nonzero updates in this particular fixture. The audit checks all 8,240 gradient tensors and 8,240 updated-parameter tensors per runtime. Maximum next-token logit differences are `1.7881393432617188e-7` on CPU and `1.1920928955078125e-7` on CUDA; causal top-1 predictions match 140/140 per device. Reward and PRM classification outputs are not counted as causal next-token predictions. PPO/RLOO scalar training losses are unmeasured/null, rather than asserted zero.

These rows use two-layer, hidden-size-32, vocabulary-256 Qwen fixtures, FP32, eager attention and SGD. They establish API and training-path breadth; they do not establish the full pretrained model's numerical parity, AdamW/AMP behavior or convergence for every method. PRM uses a token-classification model and shared initial head weights. NashMD executes its actual geometric-mixture generation wrapper; controlled replay is limited to discrete inputs, without replacing Trainer loss mathematics.

Accepted evidence: `trl-current16/final-summary.json` (SHA256 `31e04caddcdae903d79c72b64303171c6cf0ddb790ca643a0dbd4c705a89d229`), `compact-metrics-and-storage.json`, and `inventory-selected16.json`. The final candidate artifacts occupy 117,042,966 bytes across 22,704 files, outside Git. The runtime fingerprint is `3b6213b8e6ea49bdd772630b218cdf6d09188a3a5f5f90906a55bb08dcf521df`.

`tools/run_trl_gate.py` is the maintained entry point. A separate actual default three-step CPU run creates 16 fresh genuine-Torch references, executes 16 candidate processes, and audits all 3,694 observed fields, including 1,236 gradients and 1,236 parameters: exit 0, zero fallback. The source-bound receipt is `trl-current16/maintained-cpu-default3-r1/maintenance-entry-receipt.json`. There are 55 pure gate tests. A review found that matching but empty producer-source maps could pass the collector. It now requires all five method-specific producer/case/helper/runner names, with negative empty, truncated and unrelated-map tests. The tightened collector re-audits the same 16 retained pairs without changing any numerical result or manifest hash; `source-guard-followup-receipt.json` binds the later guard/provisioning files separately from the original measured entry-point sources. The new CPU workflow provisions isolated pinned environments and retains numerical/provenance artifacts. Its local entry point was executed; the GitHub workflow itself has **not** been triggered, and the remote CPU-wheel provisioning is not presented as observed CI success.

## Module precision repairs and exact continuation

Two actual execution-path defects were isolated before changing the shared owners:

- **Nested SiLU changed after warm-up.** The first nested call used the Python functional implementation, which widens low-precision inputs and rounds once. After publication, the C++ shortcut used a narrow sigmoid followed by multiplication. An actual layer-0 MLP replay matched Torch on its first call, then reproduced the uninterrupted/resumed discrepancy on the next call. The low-precision shortcut now declines to the existing functional owner; the FP32 shortcut remains. Standalone SiLU probes had missed the nesting/publication condition.
- **Frozen mixed-dtype RMSNorm returned the wrong dtype.** With a frozen Key branch, optional RMS inference fusion returned FP16/BF16 although the module's FP32 weight multiplication required FP32. The trainable Query branch already used the correct path. Before the repair, the FP16 Key normalization differed in 7,009 of 7,168 elements, up to `0.099609375`. Both the frontend admission and C++ shortcut now decline mixed input/weight dtypes, after the existing ACL special-order hook, leaving the module's original forward in charge. Standard `torch.nn.RMSNorm` retains its own output contract; native Jittor kernels and model/Trainer mathematics were not replaced.

The independent genuine-Torch controls pass all 32 low-precision SiLU output/VJP fields and all 32 mixed-RMS output/VJP fields. Controls cover autocast on/off, first/warmed calls, frozen/trainable inputs, `no_grad`, nonunit FP32 weights and output types. The actual first-layer Query/Key, rotary inputs/outputs, Value and attention mask now agree exactly with Torch in both precision profiles.

Focused results are one nested-SiLU CPU test, two mixed-RMS CPU tests, and five CUDA tests covering both new regressions plus existing same-dtype RMS controls. The CPU mixed-RMS joint receipt has three passes, of which the third is the legacy-counter regression. Evidence is `trl-local-fixes/amp-resume-final-summary.json`, `trl-local-fixes/amp6-nested-silu-after-comparison.json`, `trl-local-fixes/amp9-mixed-rms-after-comparison.json`, `trl-local-fixes/amp10-attention-after-comparison.json`, `trl-local-fixes/amp9-mixed-rms-green.log`, `trl-local-fixes/nested-silu-final-cpu.log` and `trl-local-fixes/adam-legacy-rms-after-cpu.log`.

The unchanged practical producer then completed four fresh candidate runs: ten actual updates and a separate checkpoint-5 continuation through updates 6–10, for FP16 and BF16. FP16 uses the shared initial GradScaler scale of 1024. Each runtime restores its own checkpoint; the reference for each continuation is its own uninterrupted run.

| CUDA practical SFT profile | Own-resume fields | Resume strict / byte equality | Resume max abs | Cross-runtime failing fields | Final adapter error / Torch update | Next-token logits max abs | Top-1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 2,829 | Pass / exact | 0 | 3,159 | 4.04874% | 0.205078125 | 3/3 |
| BF16 | 2,829 | Pass / exact | 0 | 3,329 | 10.17503% | 1.375 | 3/3 |

All four candidate runs complete with zero backend fallback. The checkpoint binding, RNG, scheduler, scaler and observation inventories are checked. This closes the measured AMP continuation failure; it does **not** make the cross-runtime strict comparisons pass. Matching three top-1 tokens does not establish equivalent distributions or model quality.

Adapter-update error means `L2(Jittor final adapters - Torch final adapters) / L2(Torch final adapters - shared initial adapters)`, aggregating all 112 tensors after step 10. The denominator is the actual Torch training update, not the much larger parameter norm. The shared initial adapter archive SHA256 is `25a0f1f55be18f546ef6a1da41def3a405c1f2224e0a348a65fd7083af267624`.

Use `trl-local-fixes/rmsfix-practical-run-summary.json`, `trl-local-fixes/rmsfix-practical-source-binding.json`, `trl-local-fixes/rmsfix-{fp16,bf16}-{cross,jittor-resume}-comparison.json`, `trl-local-fixes/rmsfix-*-adapter-update-error.json`, `trl-local-fixes/rmsfix-*-final-logits-diagnostic.json` and the corresponding `trl-local-fixes/jittor-cuda-*-rmsfix-*.launch.json`/`.source.patch`. Earlier `silufix-*` measurements isolate the first repair and must not be relabeled as the final two-repair source.

## What the remaining rounding evidence establishes

After both repairs, the earliest observed first-layer attention-weight difference is FP32 softmax error of `1.1920928955078125e-7`. The following FP32 RMS boundary was replayed on identical actual inputs and weights in independent runtimes; each replay reproduces its actual model output byte-for-byte.

The cross-runtime RMS difference is at most two FP32 ULPs for the FP16 profile and three for the BF16 profile. Both runtimes are within three ULPs of the correctly rounded FP64 reference; their 99th-percentile distance is at most two ULPs. Portable and ordinary execution also retain small differences, so disabling fusion does not establish exact agreement. See `trl-local-fixes/amp11-actual-rms-rounding-audit.json` and its two source-bound replay traces.

This bounds a concrete remaining FP32 arithmetic discrepancy. It is **not a proof that every final training difference is normal floating-point rounding**, nor a reason to change the numerical budget. Cross-runtime training parity remains an open numerical-kernel investigation. General AMP coverage is also not claimed; the independently observed FP32-input SiLU/autocast boundary lies outside Qwen's measured low-precision SiLU input path.

The first-step adapter audit separately follows all 112 adapters through gradients and AdamW updates. Global relative gradient differences of 0.9724% (FP16) and 3.1653% (BF16) become first-update differences of 10.9157% and 20.8215%. Among 688,128 nonzero B-gradient coordinates, 2,092 and 7,524 change sign; these explain 99.8816% and 99.9598% of first-step squared parameter error. Most of this error is not in the epsilon-dominated regime: an ideal first Adam update behaves approximately as `-lr * sign(g)` when `abs(g)` greatly exceeds epsilon. Applying the same scalar formula independently reproduces the observed pair differences to relative residual below `8e-7`.

This explains a measured amplification mechanism after gradients already differ. It neither identifies every upstream gradient discrepancy nor proves the entire ten-step trajectory harmless. The source-bound audit is `trl-local-fixes/amp-first-step-mechanics-audit-20261025-v2.json` (SHA256 `a86e7c60bce75e96007b03574f883cf8a343cd625f0ae5866b130a68b1e206c8`). It must not be conflated with the epsilon-dominated single-coordinate DDP example below.

## Adam counter lifecycle and source ownership

Normal integer Adam/AdamW step state now exposes a stable mutable scalar Tensor under the tested dtype/device policies. The focused lifecycle checks cover identity and `state_dict` aliasing, integer mutation affecting the next update, zero/replacement/deletion, parameter reorder on restore, newly added groups and retaining a counter without retaining optimizer parameters.

A separate legacy-load repair preserves an already initialized Adam state whose serialized counter is numeric zero and whose moment tensors are present. It must not be mistaken for an uninitialized slot merely because zero is falsey. Its next update and moments match the fresh control. Genuine Torch independently passes 12 CPU/CUDA controls, including the legacy case; the final CUDA counter/default-fused suite passes seven tests with no skips. Receipts are `trl-adam-legacy/independent-oracle.json`, `trl-local-fixes/adam-final-focused-cuda.{json,log}`, and `trl-local-fixes/adam-legacy-rms-after-cpu.{json,log}`.

The core run exposed a real structural failure: `optimizer_api.py` had 846 lines against its unchanged 800-line budget. `_AdamStep` and `_AdamSteps` were moved to the Adam owner, reducing the API module to 780 lines. Class text and AST are identical across the move; the existing `_update_in_target_dtype` and `_adam_step` ASTs are also unchanged. This repaired the implementation layout rather than waiving the test. `trl-optimizer-motion/motion-proof.json` retains the before/after hashes and source snapshots; `trl-local-fixes/core-torch-trl-final.log` retains the original budget failure. Source-motion proof does not itself substitute for numerical testing.

Externally injecting a fractional live counter remains outside this normal-integer TRL scope. Genuine Torch accepts a mutation to 7.5 and advances it to 8.5; the candidate's extracted bookkeeping advances it to 8. The candidate observation is an AST-extracted scalar model, not a fresh Jittor numerical run. This limitation does not invalidate normal integer lifecycle tests, but prevents a claim of complete Torch optimizer semantics. See `trl-adam-live/counter-fractional-readonly-review-20261025.json`. No general graph-capture acceptance follows from the eager lifecycle repair.

## Expanded full-model LoRA configurations

The new method-aware producer runs official SFT, DPO and GRPO Trainer mathematics with AdamW. Each row below includes ten actual candidate optimizer updates and zero fallback, against an independent same-device genuine Torch run. All six cross-runtime aggregate strict comparisons still fail.

| FP32 configuration | Device | Failing / observed fields | Final adapter error / Torch update | Final next-token logits max abs | Top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| DPO, accumulation 1, length 32 | CPU | 65 / 5,679 | 0.0160984% | 0.000312805176 | 3/3 |
| DPO, accumulation 1, length 32 | CUDA | 98 / 5,679 | 0.0669442% | 0.000242710114 | 3/3 |
| GRPO, accumulation 1, length 32 | CPU | 235 / 5,709 | 0.146975% | 0.000514507294 | 3/3 |
| GRPO, accumulation 1, length 32 | CUDA | 146 / 5,709 | 0.139517% | 0.000273227692 | 3/3 |
| SFT, accumulation 2, length 128 | CPU | 450 / 5,689 | 0.168794% | 0.000598013401 | 3/3 |
| SFT, accumulation 2, length 128 | CUDA | 65 / 5,689 | 0.0132453% | 0.0000791549683 | 3/3 |

The logits column is extracted specifically from the three complete-vocabulary `final/*/next_logits` fields, not the comparison's global maximum over unrelated losses, gradients and states. Adapter error uses the actual-update denominator defined above. These are training/configuration measurements on ten controlled text records, not accuracy evaluation.

SFT executes two actual microbatches per update, including unequal lengths reaching 128; the first observed pair is 128/39 tokens in both runtimes. The observer forwards `num_items_in_batch` unchanged to official SFTTrainer/Trainer and Qwen's loss. No replacement averaging or extra accumulation division is introduced. This tests the pinned upstream token-loss convention; it is not an independent proof of equivalence to every possible concatenated-batch objective.

DPO uses the LoRA policy with adapters disabled for its reference. GRPO retains upstream reward normalization, KL, loss and backward. Its oracle samples completions and the candidate replays only the resulting discrete tokens under checked inputs/configuration; this is controlled training-numerics coverage, not online sampler acceptance. The observed GRPO oracles have nonzero advantages in eight CPU and nine CUDA update groups, with ten generation calls each. The three final prompt logits are computed from each trained candidate rather than replayed.

Accepted receipts are `trl-lora-expanded/{dpo,grpo}-fp32-a1-l32-{cpu,cuda}-v2-comparison.json` and `sft-fp32-a2-l128-{cpu,cuda}-v2-comparison.json`, with their `*-v2/trace/manifest.json` and launch records under `trl-local-fixes/lora-*.launch.json`. The initial GRPO v1 attempt stopped before training on tuple/list protocol metadata; it remains a preserved failed attempt, not an accepted result.

The measured v2 producer, cases and comparator are archived in `trl-lora-expanded/measured-v2-sources/`, with `source-bindings.json`. Later pre-write validation and an additional mixed-RMS runtime-owner fingerprint are guard-only changes: `trl-lora-expanded/measured-v2-sources/guard-only-followup-proof.json` establishes identical measured `run` AST after removing the declared non-numeric guards. This preserves the distinction between the actual measured producer hash and the later maintained source; it does not manufacture a new numerical run.

## Actual two-GPU default DDP training

The official SFTTrainer + Accelerate + PEFT path now executes on both RTX 2080 Ti cards with NCCL, full Qwen3-0.6B and FP32 LoRA: ten actual updates, checkpoint at step five, and a separate fresh-worker continuation through steps six to ten. Rank-local inputs differ; upstream global token weighting remains enabled. All ranks agree on gradients, parameters and 336 Adam state fields, with positive updates, checked physical devices and zero fallback.

The first real attempt stopped before training because Accelerate imports the canonical DDP communication-hook namespace even for its default NO-hook option. The repair publishes that namespace transactionally so the default path can reach existing native DDP reduction. Optional compression hooks, PowerSGD and arbitrary custom-hook registration explicitly raise `NotImplementedError` and remain `Fidelity.UNIMPLEMENTED`. No compression algorithm or replacement collective was introduced. Ten focused CPU namespace/registration tests and 30 producer guard tests pass.

| Two-GPU comparison | Compared fields | Strict result / failing fields | Max abs across observed fields |
| --- | ---: | --- | ---: |
| Candidate vs genuine Torch | 5,686 | Fail / 13 | 0.00006771087646484375 |
| Candidate fresh resume vs own full run | 2,846 | Exact comparison fails / 1,350 | 0.00007343292236328125 |
| Independent candidate uninterrupted repeat | 5,686 | Exact comparison fails / 4,274 | 0.0000896453857421875 |

All three comparisons retain 3/3 final top-1 matches. Cross-runtime softmax KL is at most `2.061e-11`. The 13 cross failures are ten observations of one adapter coordinate and three final logit fields. At layer 4 `q_proj.lora_B[420,1]`, first-step gradients are `-1.1059456e-9` (Torch) versus `+1.3096724e-10` (candidate), against AdamW epsilon `1e-8`. The same scalar first-update formula predicts the resulting parameter gap of `1.12509e-5` with residual below `7.1e-13`. The whole gradient tensor's relative L2 difference is `5.7344e-6`; this localized example does not indicate a large global gradient error or establish the upstream cause.

Resume inputs, RNG, scheduler and checkpoint bindings are exact. Entire observed steps six and seven are exact; the first difference appears in step-eight gradients (`1.0430813e-7`) while both local losses remain exact. The independent uninterrupted repeat exhibits a larger difference without checkpoint restoration, demonstrating a repeat-variation baseline. It does not establish the root cause of all discrepancies. A secondary check using the ordinary cross-runtime numerical budget passes training states in both resume and repeat comparisons, but still fails their three final-logit fields. Neither the original exact checks nor this secondary budget is relabeled as passing.

Accepted receipts are under `trl-ddp-20261025-v2/measurements/`: `jittor-retry1/{ddp-local-receipt,near-zero-gradient-adam-amplification,resume-standard-budget-diagnostic}.json`, and `jittor-repeat1/{repeat-local-receipt,repeat-comparison,repeat-standard-budget-diagnostic,final-logit-budget-failures}.json`. Together the measured arrays, checkpoints and receipts occupy 1,008,441,889 bytes, outside Git. These rows do not admit AMP DDP, FSDP, ZeRO, multiple nodes or communication compression.

## Matched full-model SFT throughput

The throughput producer measures a separate official SFTTrainer path on one otherwise idle physical RTX 2080 Ti, using identical full pretrained Qwen3-0.6B, initial base/buffers/adapters, package pins, eager attention, four CPU threads and strict/TF32-off math. It uses the same LoRA/AdamW settings as above, batch one, accumulation two and a constant learning rate. Ten sequence lengths repeat as `[128,39,128,40,128,40,128,39,128,40]`. Logging, save and evaluation callbacks are disabled; upstream SFT token-accuracy metrics and normal in-loop input preparation/H2D remain included in both runtimes. The first timed accumulation group is prefetched and transferred to CUDA before `on_step_begin`, including DataLoader lookahead. This is therefore a synchronized Trainer-update interval rather than complete end-to-end loading/H2D latency; subsequent in-loop data loading/H2D remain included.

Separate 25-update priming workers precede the retained experiments. Each retained worker then has five warm-up updates and twenty timed actual optimizer calls, processing 40 sequences, 3,352 nonpadding tokens and 3,312 causal prediction targets. Real CUDA synchronization brackets the timing interval. The native optimizer's public state is not read or exposed until timing stops; there are no per-step observer tensor copies or gradient hooks. Stock Trainer scalar metrics and subsequent parameter-dependent forwards force the intermediate updates, and final synchronization completes the last one. After timing, every one of 112 trainable gradients is independently checked finite, nonzero and physically on CUDA; all counters must equal 25, moments must be finite, and the frozen base must be unchanged.

Three repeats per runtime/profile alternate runtime order on the same GPU. Any new or changed Jittor shared object during a retained interval rejects that row. Hardware clock/power configuration is recorded rather than changed. Library autotuning has no comparable public counter, so binary-cache stability is not presented as proof of zero library tuning. Setup and warm-up use existing caches and are explicitly not genuine cold-start measurements. SM75 BF16 execution does not imply native BF16 tensor-core acceleration. No allocator-based VRAM comparison, convergence or cross-runtime numerical admission follows from these timings. The maintained producer is `trl_throughput_probe.py`; raw evidence stays under `trl-throughput-20261025-v2/`.

All 18 retained workers complete with 25 actual updates each (five warm, twenty timed), finite/nonzero gradients in every adapter, unchanged frozen bases, no fallback and no changed/new JIT binaries during timing. Median and full three-repeat range below refer to the twenty-update interval; token rates count nonpadding input tokens.

| Precision | Torch seconds: median [min, max] | Jittor seconds: median [min, max] | Torch tokens/s | Jittor tokens/s | Jittor / Torch throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 5.9208 [5.9172, 5.9422] | 6.3726 [6.2892, 6.3852] | 566.14 | 526.00 | 0.9291 |
| FP16 | 7.0377 [6.9738, 7.0427] | 8.1962 [8.1269, 8.3264] | 476.29 | 408.97 | 0.8587 |
| BF16 | 6.8135 [6.7717, 6.9886] | 10.2801 [10.1596, 10.4349] | 491.97 | 326.07 | 0.6628 |

Jittor is slower in all three measured profiles. These data establish a reproducible baseline for this short-sequence, accumulation-two LoRA workload; they do not claim a general model/framework speed ratio. In particular, completing mixed-precision training quickly does not close the separate strict numerical failures. The unchanged measured results are in `trl-throughput-20261025-v2/summary.json` and `measure-*/result.json`.

The independent audit confirms identical 423 initial tensor hashes and 1,269 runtime-owner hashes across all 18 rows, plus a stable compiled-core binary per runtime. Exact measured producer/test sources are preserved under `trl-throughput-20261025-v2/measured-source/`; the measured producer SHA256 is `e831e71efd6ba5eba0e719d929a8d9121554c1f78f855398bd1050b3e8af5815`. A later collector-only repair rejects nonfinite durations and inconsistent runtime-source/core bindings. Its 26 pure tests and `collector-guard-proof.json` establish identical worker, driver, timer/training, scheduling and token-count source/AST, with all 18 original rows re-audited and numerical profile summaries unchanged. `throughput-local-receipt.json` binds rows, binaries, sources and `summary-hardened.json`; original `summary.json` remains untouched.

## Repository verification

The final Torch-mode structural suite passes **1,516 tests, four skips**, with no layout/budget waiver (`trl-local-fixes/structure-trl-final-r2.{json,log}`). Native CPU core checks pass 104 tests, with 41 skips and one expected failure; the frozen Torch CPU core plus focused regressions pass 243 tests with one skip. These are separate process modes. The CUDA Adam lifecycle/default-fused suite passes seven tests, and the precision-focused CUDA suite passes five.

The final combined pure-tool suite passes **129 tests** (`trl-local-fixes/trl-final-tools-r3.log`), covering trace completeness, source binding, checkpoint guards and timing protocol rejection cases. New Python files pass Ruff; changed existing owners introduce no additional diagnostics relative to HEAD (`trl-local-fixes/trl-final-ruff-owner-differential.json`). After collector-only hardening, a focused repeat of documentation, packaging and pytest-collection contracts passes 57 tests (`trl-local-fixes/structure-trl-guard-final.{json,log}`). Layout and `git diff --check` pass. Skipped tests are not counted as device coverage, and these repository checks do not override any numerical failures recorded above.

## Machine limits and remaining work

The available two RTX 2080 Ti devices and 80-GiB container memory budget suffice for the measured text fixtures, complete-model LoRA and FP32 two-rank DDP paths. The data disk still has approximately 47 GiB free at the closing checks; no new disk-capacity failure is observed. That does not remove the [earlier full-parameter FP32 PPO limit](https://github.com/Jittor/jittor/blob/2.0-refactor/docs/results/2026-10-23-trl-full-and-practical.md#hardware-limits): its unmodified single-device placement needs 14,305,210,368 bytes before activations/context/workspaces, beyond one card's 11,347,623,936 physical bytes. A sharded, offloaded or LoRA PPO configuration would be a separately defined experiment; two cards do not by themselves validate it.

Remaining work includes the recorded full-model strict numerical failures, unsupported custom communication options, unmeasured model/Trainer configurations and other hardware backends. The numerical investigation can continue on this machine; it is not described as solved or as requiring a larger GPU merely because bounded functional tests passed. Diffusion Trainer exports, remote CI execution and multiple-node/backend acceptance remain outside the measured Qwen text scope.

## Reproduction and acceptance boundary

Use isolated candidate/oracle interpreters, offline model/data artifacts and exclusive per-process/device `JITTOR_HOME` caches as in the [original protocol](2026-10-23-trl-full-and-practical.md). The maintained text gate is `tools/run_trl_gate.py`; individual practical/LoRA, DDP and throughput entry points live under `agent/skills/torch-compat-repo-runbook/scripts/` and are documented in its runbook. Practical correctness uses `trl_practical_probe.py`/`trl_practical_compare.py`; expanded methods use `trl_lora_probe.py`/`trl_lora_compare.py`. Reproduce measured v2 rows with the archived measured sources and exact launch arguments; use current sources for new, separately labeled measurements. `trl_forward_boundary_probe.py` is a diagnostic observer that intentionally stops after a selected forward, not an accepted substitute for a complete training/continuation pair.

The accepted changes repair concrete runtime contracts and demonstrate complete bounded runs. They do not establish whole-library coverage, universal determinism, full optimizer/AMP equivalence, strict distributed numerical parity or relaxed numerical budgets. Subsequent acceptance must retain actual update counts, complete adapter/gradient/state inventories, physical device/fallback guards, independent references and unchanged comparison budgets.
