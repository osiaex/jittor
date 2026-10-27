# Full-checkpoint TRL methods and practical SFT configurations

- Status: Bounded validation complete; numerical and capacity gaps remain.
- Reviewed: 2026-10-23.
- Baseline: `8f894ca47c9c4932cfcc6d164cd6faae0c1e6c75`, synchronized against
  `2.0-refactor` at `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`.
- Owner: Torch compatibility and downstream validation maintainers.
- Recheck when: upstream pins, numerical kernels, model, optimizer, precision,
  serialization or these diagnostic protocols change.

The selected eight-method full-checkpoint subset completes on **CPU 8/8** and
**CUDA 5/8**. CUDA Online DPO and RLOO fail in Jittor's SGD state placement while
their matched Torch runs complete; full FP32 PPO exceeds this card's basic memory
capacity. All **13 completed pairs** retain strict numerical failures. All final
parameter-value tensors pass their absolute budget, and every completed causal
scenario has next-token top-1 agreement on its ten fixed prompts; those checks do
not establish update-delta agreement or training quality. Formal L2 admission
remains unclaimed.

The four practical LoRA/AdamW SFT profiles also complete, with cross-runtime
numerical failures in each. Reproduced AMP dtype-policy and Adam counter-state
representation gaps are tracked alongside the SGD memory gap in the issue ledger.

## Scope

Two independent axes extend the [earlier tiny-model breadth matrix](2026-10-22-trl-mainstream.md):

1. Full pretrained Qwen3-0.6B, FP32 SGD, targeting ten optimizer calls through KTO, ORPO, CPO,
   Reward, PPO, RLOO, GKD and Online DPO. Earlier full SFT/DPO/GRPO evidence
   remains separate; this is the remaining eight-method subset.
2. Official SFTTrainer on the same full checkpoint with PEFT LoRA, AdamW,
   CPU/CUDA, FP32/FP16/BF16 controls and fresh-process checkpoint continuation.
   These SFT configurations do not establish support for those options across
   every TRL Trainer.

The pins remain TRL 0.21.0 at `46d09bd2408f17605409fb3ee8ba12705add7faa`,
Transformers 4.56.2, Accelerate 1.10.1, datasets 3.6.0 and genuine Torch
2.8.0+cu128, with NumPy 1.26.4 and Python 3.11.17. Practical SFT additionally
uses PEFT 0.17.1 in both isolated
interpreters. Installing that package did not replace Torch or Transformers.
The checkpoint is Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca` with shared original RoPE buffers.

Cross-runtime floating-point observations use the existing strict budget
`atol=1e-5, rtol=1e-4` and field-scale floor. Discrete inputs and scalar counter
values use exact comparison; own-runtime checkpoint continuation is also exact.
Completion, zero backend fallback, strict numerical agreement
and same-runtime exact checkpoint restoration are reported separately. These
bounded experiments are not a convergence, throughput or general-library gate.
Recorded training runtimes include full-tensor copying, hashing, norms and trace
comparison; candidate and oracle instrumentation costs differ. They must not be
used as a fair training-throughput ratio.

## Full-model measurement protocol

The eight-method probe loads the complete 28-layer pretrained checkpoint and
its official tokenizer. The fixed setup uses eager attention, four CPU threads,
SGD learning rate 1e-4, zero weight decay and a constant learning-rate schedule.
Every model role's initial parameters and buffers are
hashed and matched independently. Classification heads are deterministic shared
fixtures; GKD's teacher uses the checkpoint with its final RMSNorm weight scaled
by 1.01 to ensure a nonidentical distillation target. These reward/value/teacher
fixtures test execution and numerics, not reward quality or distillation quality.

There are ten short examples. KTO uses batch two and visits twenty examples over
ten optimizer calls. RLOO generates two completions per prompt (twenty
sequences over ten calls). Online algorithms retain their official generation and training
math, with oracle token-choice replay so both runtimes train on identical
rollouts. PPO trains both its complete policy and complete value model. PPO and RLOO
each visit ten prompts in five rollout batches, with two optimizer minibatches
per rollout and up to four generated tokens per response.

Each update checks all trained gradients for presence/finiteness and records
full-tensor norms, embedding-output gradients, and one element per 4096 entries
of parameter gradients and updated parameters. The last update compares **all
parameter-gradient and final-parameter elements**. Final fixed-prompt logits
are also compared. PPO/RLOO do not directly record a scalar loss comparison;
those methods must not be described as having direct scalar-loss parity.
Final-parameter comparisons use parameter values, not error relative to the
actual training update, which can be much smaller. Intermediate samples must not be described as full all-step
gradient parity. Candidate bulk arrays are stream-compared and represented by
hashes/statistics, without duplicate retained payloads. No L2 admission is claimed.

An initial harness revision compared integer-key config dictionaries with their
JSON-decoded string-key equivalents. The guard rejected the candidate before
training. That harness defect was fixed through canonical JSON serialization;
new independent source-matched pairs are used. Superseded unpaired runs are not
accepted numerical evidence.

The accepted full-model producer is bound to SHA256
`d7bddf0a6a6bba92d3e81c7cecae97e45abbf0fa288623ca3a4edf81d7f3574d`
(`trl_full_probe.py`) and
`0e500d15e462b4c17bbcdb413001c6d821054a2e1f95c10807350f77ab10e729`
(`trl_full_cases.py`). Collector hardening adds required metric inventories and
archive verification without changing any training computation or threshold.


## Fixed FP32 SGD matrix

Each completed row has ten optimizer calls, complete final parameter gradients
and parameter values, and zero backend fallback. “Fail” is the unchanged strict
numerical result over the recorded observations, not a runtime exception.
Final top-1 is computed independently from each trained model; rollout token
choices during training are replay-controlled. Output error means
all-vocabulary next-token logits over ten prompts, except
Reward, where it means ten scalar scores. Top-1 agreement is not a quality test.

| Method | Device | Final grad max abs | Final grad relative L2 | Final grad failing tensors | Output max abs | Top-1 | Strict result / limitation |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ORPO | CPU | 0.000976 | 0.000111 | 177/310 | 0.00186 | 10/10 | Fail |
| ORPO | CUDA | 3.81e-05 | 3.35e-06 | 0/310 | 3.62e-05 | 10/10 | Fail |
| CPO | CPU | 0.00224 | 0.000326 | 282/310 | 0.00148 | 10/10 | Fail |
| CPO | CUDA | 8.23e-05 | 1.41e-05 | 6/310 | 8.01e-05 | 10/10 | Fail |
| KTO | CPU | 0.000217 | 0.000157 | 112/310 | 0.00226 | 10/10 | Fail |
| KTO | CUDA | 7.43e-05 | 5.83e-05 | 40/310 | 0.00285 | 10/10 | Fail |
| Reward | CPU | 7.52e-05 | 1.66e-05 | 18/311 | 3.49e-05 | n/a | Fail |
| Reward | CUDA | 0.000133 | 2.73e-05 | 62/311 | 7.41e-05 | n/a | Fail |
| GKD | CPU | 1.65e-07 | 8.5e-05 | 0/310 | 0.000132 | 10/10 | Fail |
| GKD | CUDA | 3.28e-07 | 0.000175 | 0/310 | 9.11e-05 | 10/10 | Fail |
| Online DPO | CPU | 0.000429 | 0.000104 | 136/310 | 0.00478 | 10/10 | Fail |
| Online DPO | CUDA | — | — | — | — | — | Jittor OOM; Torch completes 10 |
| RLOO | CPU | 1.76e-05 | undefined | 3/310 | 0.00275 | 10/10 | Fail |
| RLOO | CUDA | — | — | — | — | — | Jittor OOM; Torch completes 10 |
| PPO | CPU | 0.00403 | 0.000625 | 284/621 | 0.00382 | 10/10 | Fail |
| PPO | CUDA | — | — | — | — | — | VRAM lower bound; unrun |


RLOO CPU has an additional measurement boundary: genuine Torch's optimizer
calls 6, 8 and 10 receive exactly zero total parameter gradient and make no
parameter change. Jittor's calls 6 and 8 are also zero; call 10 has gradient
L2 1.46264e-4 and update L2 8.315e-9 (96,596 changed elements). Its maximum
final-gradient discrepancy is 1.76466e-5, with four elements outside budget.
Since the reference final-gradient norm is zero, its relative L2 is **undefined**,
not zero. Ten optimizer calls must not be described as ten nonzero parameter
updates. The source of this final zero-versus-small-gradient difference has not
been isolated; final parameter-value agreement does not resolve it.

## Hardware limits

Both devices are RTX 2080 Ti (SM75), each with 11,347,623,936 physical bytes.
The unmodified full-parameter FP32 PPO placement needs four complete models plus
two complete gradient sets: **14,305,210,368 bytes before activations, context
or workspaces**. That configuration is capacity-blocked on one card. CPU PPO
remains a separate applicable test; shrinking the model would not satisfy this
full-checkpoint cell.

The host-memory display is not the process budget: the container has an
**80 GiB cgroup memory limit**. The initial Jittor CPU PPO attempt received
SIGKILL before its first optimizer step while other training processes were
active. The cgroup recorded historical maximum usage at its limit, with substantial tmpfs
trace storage charged as shared memory. Kernel logs were inaccessible and no
pre-run failure-counter baseline existed, so this is evidence consistent with
memory pressure, not a proven kernel OOM attribution. The failed attempt remains
separate from the monitored isolated retry. That retry uses the same frozen
default CPU protocol and completes all ten optimizer calls with exit zero, no
resource-guard interruption and no increase in the cgroup failure counter.
Sampled peak cgroup usage is 79,297,810,432 bytes (73.85 GiB). This establishes
execution under the isolated resource envelope; the original SIGKILL remains
an explicitly separate incident.

The default Jittor CUDA Online DPO attempt reaches its first SGD step and
fails allocating a 12 MiB optimizer-state device copy, with 8 MiB reported free.
The corresponding genuine Torch run completes ten steps. Native SGD constructs
full-sized `values` momentum buffers even with momentum zero, and `pre_step`
aligns these buffers before taking the momentum-free update branch. The failing
alignment excludes gradients from its state-buffer loop. Those additional
buffers account for 2,384,199,680 bytes for this policy; if all are materialized,
three model roles plus policy gradients plus these buffers require approximately
11.921 GB before activations, exceeding the 11.348 GB device. This is a concrete
memory gap and a conditional allocation explanation, not a floating-point error
or a proof that every scheduler configuration must fail.

Default CUDA RLOO also fails before the first completed optimizer update in the
same alignment path, while requesting 8 MiB. Its genuine Torch counterpart
completes ten steps. A separately recorded Online DPO control sets both
`JT_AUTO_FLUSH_OPS=0` and `JT_ASYNC_FLUSH_OPS=0`; the effective settings are
confirmed by the runtime environment report. It still fails on the same 12 MiB
state copy. Therefore this tested scheduling change is not a demonstrated
workaround. Neither failed candidate contributes a ten-step CUDA numerical
comparison. No runtime implementation was changed to conceal these failures.

The device reports native BF16 support as false, but support including emulation
as true. Genuine Torch CUDA BF16 matmul/backward and autocast Linear/backward
both execute with finite values on the actual CUDA device, as do FP16 controls.
Therefore lack of native BF16 acceleration is not labeled functional BF16
unsupported. Actual full-model Trainer outcomes are evaluated separately.

## Practical SFT protocol

LoRA targets q/v projections, rank 8, alpha 16, dropout zero:
**1,146,880 trainable adapter parameters in 112 tensors**. The full pretrained
base remains frozen and its hashes are checked. AdamW and official Trainer
checkpoint routines execute normally, using ten distinct sequential texts.
AdamW uses learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8 and weight decay
0.01, with a linear schedule, no warmup, batch one and maximum length 32.
All adapter gradients, updated parameters and optimizer states are recorded.

An uninterrupted ten-update run saves checkpoint 5. A new process loads that
checkpoint and completes updates 6–10, preserving the original ten-update
scheduler configuration. Exact continuation checks cover gradients, parameters,
Adam states, final logits, input sequence, RNG, scheduler and GradScaler state.
This is same-runtime restore, not cross-framework checkpoint interchange.

Torch's live Adam step counter is a scalar FP32 Tensor; Jittor exposes a Python
float, which the trace's NumPy conversion records as FP64. Counter values are
compared exactly after explicit scalar-only canonicalization, while the live
representation difference remains reported and prevents strict API parity.
All other dtype and numerical checks remain unchanged.

A 2026-10-24 audit corrected the earlier claim that the checkpoint itself stored
FP64 counters: all four original checkpoint-5 files contain 112 FP32 scalar
counters each, all equal to five. Jittor's state-dict serializer already creates
FP32 counters separately from its live Python floats. Independent live-state
probes also expose missing Tensor identity/aliasing and in-place update behavior;
this API issue remains open as KI-COMPAT-019. The unversioned evidence is in
`$JITTOR_LAB_ROOT/_state/trl-local-fixes/adam-counter-existing-checkpoint-audit.json`
and `adam-counter-{torch-cpu,torch-cuda,jittor-cpu}-v2.json`. This correction
changes the cause and scope of the representation issue, not any measured
training errors or continuation results.

## Practical SFT results

All four explicit profiles complete ten actual AdamW updates in both runtimes,
with unchanged frozen-base hashes and zero fallback. All four fail the unchanged
cross-runtime strict numerical budget. The table uses the complete next-token
vocabulary at three fixed prompts; top-1 agrees at all three for every profile.
That small argmax check does not establish output distribution or training quality.

| Profile | Next-logit max absolute error | Next-logit relative L2 | Final adapter error / oracle update L2 | Jittor exact resume | Resume max absolute error |
| --- | ---: | ---: | ---: | --- | ---: |
| CPU FP32 | 3.3331e-4 | 1.7342e-5 | 0.3121% | Pass | 0 |
| CUDA FP32 | 3.2187e-4 | 1.4122e-5 | 0.1764% | Fail | 7.8559e-5 |
| CUDA FP16, initial scale 1024 | 0.3359375 | 0.0157063 | 5.1818% | Fail | 0.23828125 |
| CUDA BF16 | 1.498046875 | 0.0671191 | 15.8723% | Fail | 1.8125 |

Genuine Torch passes exact fresh-process resume in all four profiles. Jittor's
CUDA FP32 checkpoint-5 adapter tensors and Adam fields match the recorded state
exactly, but subsequent execution is not bitwise reproducible. A separate fresh
uninterrupted Jittor CUDA FP32 repeat also differs (maximum 6.819e-5), despite
identical RNG/scheduler state. This shows execution nondeterminism exists; it
does not prove that every resume discrepancy, especially AMP, is harmless.

The default FP16 scale 65536 diagnostic produced only nine real Torch updates
and five Jittor updates because GradScaler skipped overflowed steps. These runs
are retained but excluded from the ten-update comparison. A separately declared
shared scale of 1024 gives ten updates without overflow on both sides. Low
precision q-projection outputs were checked on the actual CUDA device.

A posthoc independent inventory audit confirms all eight full/resume run pairs
contain the complete 5,649 / 2,829 observations, state-step keys and matching
checkpoint-5 file hashes. The reusable comparator now rejects equally missing
fields on both sides, empty RNG/scheduler/scaler coverage, incorrect Adam steps,
unbound checkpoints and same-directory comparisons. Re-auditing four
cross-runtime comparisons and eight own-checkpoint continuations with these
stronger guards leaves every numerical-field result unchanged. Measured v1/v2
producer sources remain archived; the current training function's AST matches
measured v2, and separate receipts bind the posthoc comparator source.

Each full practical comparison also records 1,120 Adam counter representation
mismatches (Torch scalar FP32 tensor versus Jittor scalar FP64 tensor). Equal
integer values after the documented canonicalization do not remove that API gap.

## Evidence and tools

Unversioned artifacts live in `$JITTOR_LAB_ROOT/_state/trl-full/` and
`$JITTOR_LAB_ROOT/_state/trl-practical/`. The former contains native-device
capability measurements, the PPO capacity calculation, source-bound run traces
and independent pair audits. The latter contains LoRA/AdamW traces, official
checkpoints and exact continuation comparisons.

Exact independent-interpreter commands and environment controls are in the
[Torch compatibility runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md#full-checkpoint-trl-method-matrix-with-bounded-trace-storage).
Reusable scripts in `agent/skills/torch-compat-repo-runbook/scripts/`:
`trl_full_probe.py`, `trl_full_cases.py`, `trl_full_collect.py`,
`trl_practical_probe.py`, `trl_practical_compare.py`, `trl_amp_precision_probe.py`
and `trl_trace_archive.py`. Full-model collectors
validate complete trainable inventories and distinguish full final gradients
from intermediate samples. Lossless archive receipts identify compressed bulk payloads without altering
original measurement manifests; the collector verifies recovered arrays.


## Artifact storage boundary

Superseded and completed oracle bulk arrays are preserved as lossless
Zstandard archives. Every original array byte remains recoverable. The archiver checks
original array metadata, verifies decompressed file and array hashes, durably
records recovery receipts, and only then removes the redundant uncompressed
copy. Original manifests remain unchanged; archive journals and completion
receipts describe current storage. Old superseded runs remain recoverable, and
no previous-round data is discarded.

New CPU oracle traces use a private 40 GiB tmpfs selected by
`TRL_ORACLE_TRACE_ROOT`; selected CUDA oracles use container-local temporary
storage. Persistent run directories contain trace symlinks and storage receipts.
**RAM arrays, including lossless archives there, do not survive container shutdown
or reboot.** Container-local temporary files also require preservation before
container cleanup. Launch logs, pair audits, manifest snapshots and initialization
extras are copied to persistent storage. These metadata copies do not substitute
for complete durable raw-array backups. Reward CPU now has a fully hash-verified durable copy; its redundant RAM
payload was retired and both old trace references resolve to that durable copy.
The remaining RAM/temporary traces retain the lifetime limits above. No complete
persistent archive of the entire matrix is claimed.

On 2026-10-24, authorized cleanup removed large payloads from twelve superseded
TRL runs, including the four unpaired v1 CPO/ORPO oracles. Those old arrays are
no longer recoverable from their retained historical archive receipts. Original
manifests, logs and small observations remain, with per-run
`large-array-cleanup.json` markers. The formal r2 references were not pruned.
The unversioned deletion inventory and completion receipt are under
`$JITTOR_LAB_ROOT/_state/trl-cleanup-20261024/`.


## Reproduced AMP precision-policy gap

A separate identical-input CUDA RMSNorm microprobe reproduced a precision-policy
difference in both FP16 and BF16 autocast, with zero backend fallback and actual
device checks. Both runtimes leave `x.float()` in FP32. Torch retains FP32 for
the subsequent square, mean, epsilon addition and reciprocal square root; the
Jittor path computes those observations in the requested low precision. Two
FP32 operands multiplied under autocast are similarly lowered by Jittor.
The final weighted output can return FP32 in both runtimes while already
containing lower-precision error: maximum differences were approximately
0.00508 in FP16 and 0.01584 in BF16 on the fixed microprobe.

This is a reproduced AMP policy deviation, not evidence that the original FP32
rounding explanation applies to low-precision training. Existing compat AMP
source already marks its global-register policy `Fidelity.APPROXIMATE`; the
full Qwen experiment also shows larger AMP training discrepancies. The fraction
of those discrepancies caused by this specific RMSNorm path has not been isolated. Runtime implementation was
not changed while the separate full FP32 matrix was being measured. Raw
`amp-micro-*` traces and `amp-micro-comparison.json` retain the original
observations. Maintained `amp-portable-*` runs and
`amp-portable-comparison.json` independently reproduce maximum errors
0.00508272648 / 0.01584029198, with complete zero-fallback manifests and explicit
producer/runner source hashes.


## Tool and repository validation

No runtime or kernel implementation changed in this follow-up. New measurement,
comparison and lossless-archive tools pass **80 targeted tests**,
including missing observations, altered metadata, archive corruption and shared
payload paths. Repository structure validation completed with **1,516 passed,
four skipped** (two accelerator conditions and two declared skips); no unexplained
skip was recorded. The isolated CUDA JIT cache was used with CPU execution for
this required check, separately from the active CPU training cache. Its first
oneDNN build contributes to the 676-second elapsed time; this is not a training
performance measurement. Final tracked-file path, documentation and packaging checks add 27 passing
tests after the new files enter the Git index. Layout and generated-manifest
checks also passed.
