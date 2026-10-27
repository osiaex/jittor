# TRL full-checkpoint numerical alignment and local validation

- Status: measured full-checkpoint FP16/BF16 SFT and CPU smoke accepted; matched throughput measured; historical CUDA error-719 first-fault attribution remains open.
- Date: 2026-10-26 UTC.
- Baseline: `e17ddf2b666ecf6127a331c72352174edd5d5b11`; synchronized `2.0-refactor` at `ca56aa802c2acf9a22bca32d1bee3276381ed203`. Exact working sources are bound by the retained source maps.
- Runtime commit: `e7d070b0c039acc6d19eee5f81b3dfc73a4a84bb`; measured runtime/producer bytes are unchanged from the source-frozen runs.
- Owner: native numerical, executor, Torch compatibility and downstream validation maintainers.
- Review when: arithmetic, backward construction, module fast paths, optimizer updates, the pinned dependencies or the measured protocol changes.

## Full-checkpoint numerical result

The unchanged practical producer runs official TRL 0.21.0 `SFTTrainer` and PEFT
0.17.1 on all 28 layers of Qwen3-0.6B. The base stays FP32; rank-8 q/v LoRA
adapters are trainable. Ten distinct examples run sequentially for ten AdamW
updates: learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
`0.01`, linear scheduler, batch one, maximum length 32 and eager attention.
FP16 starts with loss scale 1024. Both runtimes load the same independently
exported initial adapters and buffers. Independent binary PyTorch is
`2.8.0+cu128`; Transformers/Accelerate/datasets/NumPy are
`4.56.2`/`1.10.1`/`3.6.0`/`1.26.4`.

On the measured SM75 CUDA device, all four comparisons pass the **original**
collector and budgets. Every retained array, including signed-zero bits, matches.

| Precision | Independent Torch cross-runtime fields | Fresh-process checkpoint-5 continuation fields | Failed fields | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 5,649 | 2,829 | 0 | 0 |
| BF16 | 5,649 | 2,829 | 0 | 0 |

This closes KI-COMPAT-018 for the measured protocol. The full-run inventories include ten losses, all 1,120 parameter gradients,
1,120 updated parameter tensors, 3,360 optimizer-state tensors, exact inputs and
three complete final vocabulary-logit tensors with their next-token vectors.
Each continuation independently loads its own hashed checkpoint five and
executes updates six through ten. All four producers report zero fallback.
The final adapter error divided by the actual Torch update is zero for both
precisions, replacing the earlier 3.737774%/8.396626% discrepancies. Exact
observations on these fixed examples establish this protocol's numerical
acceptance; they do not measure downstream task quality.

Evidence is retained outside the checkout under
`$JITTOR_LAB_ROOT/_state/trl-completion-20261026/numerical/formal-r7/`, including
all four manifests, arrays, checkpoints and original comparison outputs.
`formal-runtime-sources.json` binds 1,279 files with SHA256
`07fff4528a72efb614f4b003274189f503faa159c1a07ac8ea7725784555df9b`;
`formal-producer-sources.json` binds 33 files with SHA256
`23240b2db373fae0aa403cacbf175c3485aeba912cb14ab9bad9f3da8d2c28ca`.
The producer and comparison code retain their original numerical budgets.

## Shared numerical repairs

The complete result required several independently reproduced corrections:

- Narrow SiLU uses one FP32 opmath division and one public-boundary rounding;
  its VJP preserves the CUDA FMA order and connected higher derivatives.
- Weighted RMS statistics use the matched row-reduction tree and reciprocal
  square root. Its direct and variance derivative edges remain separate, so a
  residual contribution can accumulate between them. A single-edge square
  keeps the primal multiplication and the correct derivative structure.
- The generic native `gradient_accumulation_order` option preserves registration
  order at its default zero. Option one visits newer consumers first; the two
  Torch backward entries select it in exception-safe scopes. Cancellation,
  nonleaf hooks, higher derivatives and scope restoration have regressions.
- Short-row softmax and large-vocabulary CE retain the relevant FP32 reduction,
  exponential and VJP rounding boundaries. CE keeps bounded row statistics.
- CUDA FP32 powers explicitly use FP32 operands, including weak Python scalar
  bases. This independent defect was verified separately; the full-model
  protocol already supplies shared RoPE buffers.
- Shared fused AdamW retains double-precision hyperparameter calculations until
  conversion to the computation dtype, with the matching lerp FMA, staged
  state rounding and divide/addcdiv order. Independent replay of actual Torch
  gradient streams verifies all 120 parameter/moment fields byte-for-byte.
- The cached C++ RMS inference shortcut respects the precise owner's priority.
  Before repair a cold prompt matched while warm prompts bypassed that owner;
  the repaired cold/warm outputs all match the independent oracle.

Actual-input local tests supplement the complete run. Intermediate observers
are admitted only after checking their parameter-gradient hashes against an
unobserved control. Exploratory counterfactuals and earlier failed runs remain
retained with their actual source identities.

## CUDA generation and stream ordering

Three shared defects have separate causal red/green evidence. The tiled CUB
scan now has uniform collective participation and synchronized shared carry:
its direct racecheck control changes from 72 hazards to zero; official PPO
changes from 27 hazards to zero. Imperative device copies and graph launches
now publish their compute-stream work through the shared handoff scope. Legal
controlled delays previously made a subsequent worker copy read stale bytes;
after repair it reads the new values while the original launch still returns
asynchronously. Migration acquires the executor entry before the alias-ring
lock, and the original copy GIL/exception-restoration contracts remain covered.

The historical error-719 first fault remains unattributed. A static generation
call path can reach the copy owner, and default graph replay can reach graph
launch; identity operations can also avoid a copy. These observations do not
identify which defect caused the historical out-of-range index. KI-COMPAT-021
retains that distinction and the original failed trace. A passing generation
retry is not a first-fault capture. The historical fault record keeps the
index, dimension and bound, but no owning launch or allocation generation.
Its launch ring had overwritten 985 entries, and the failed candidate manifest
contains no completed arrays or generation calls. The scalar arange read was
the synchronization that exposed the failure. Those retained records cannot
recover the missing writer-to-reader history.

A separate diagnostic reconstructs all 1,277 historical runtime files from
`e17ddf2b666ecf6127a331c72352174edd5d5b11` and uses the same historical
producer bytes and original binary-Torch sampling/initialization trace on both
old and repaired endpoints. Both complete all three updates, three generations
and 372 original-protocol fields with zero fallback; their source guards pass.
This historical producer differs from the current sixteen-method producer, so
these 372 fields are kept separate from the current gate's 375 PPO fields.

Both full-print memcheck runs still return code 86: every reported diagnostic
is a handled `cuKernelGetFunction` API-400 query (196 on old, 210 on repaired).
No reported category is an illegal memory read/write, out-of-bounds index or
error 719. This is not a clean sanitizer acceptance and does not identify the
historical first fault. The initial capped-print old run, initialization waits,
cache-tool restart and interrupted compilation attempts retain their actual
statuses and source boundaries; none is substituted for the completed runs.
The unversioned `indexing/historical-ppo-pair/fullprint-pair-audit.json` binds
the final full-output comparison. Startup compilation and sanitizer wall time
are excluded from the throughput conclusions below.

## Repository verification scope

Focused checks cover the shared numerical changes, legal async copy/graph
handoffs, loader/replay/FSDP lifecycle, ring-buffer shape encoding, dense/sparse
addition, tensor copies, PEFT layer initialization, FFT layouts, repeat storage,
factory gradients and optimizer parameter leaves/aliases. The complete original
25-method PEFT node trains, saves and reloads successfully. Eager optimizer writes
keep the original allocation, live aliases and trainable leaves. A refused step
capture exits graph retention before subsequent imperative stores; accepted CUDA
capture retains its recorded state-update path and real graph replay. The
original CPU/CUDA compile contracts pass without substituting eager execution
for a required capture.

The smoke runner keeps its original selection and four-worker policy. Its
collection ledger merges worker metadata at the controller and rejects truly
empty files and unexplained all-skip files. Real two-worker positive/negative
controls verify those boundaries; neither execution requirements nor original
case assertions are disabled. Exact known opt-in reasons are tied to their
actual selector: selecting that profile withdraws its skip explanation. Real
two-worker negative controls still fail for an empty file, an unknown skip,
an enabled profile that executed nothing and a missing mandatory dependency.
A CUDA-only SwiGLU test uses the public capability requirement so its CPU skip
retains the actual backend identity; its three CUDA numerical assertions stay
unchanged. PEFT BOFT's external failing-extension context
leaks `CC`/`CXX`; its test fixture restores those two keys before later C++
contracts execute. The installed dependency remains unchanged.

The pre-existing native CUDA Tape exact-repeatability failure is independently
reproduced on clean upstream and retained. It is separate from CPU smoke and
from the measured full-model protocol.

The maintained CPU smoke selection passes in both original process modes:

| Session | Passed | Skipped | Expected failures | Non-strict XPASS | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 4,723 | 2,123 | 18 | 1 | 0 |
| Torch | 4,619 | 1,793 | 1 | 0 | 0 |

The Native result is the actual successful phase of `smoke-final-r7`; its
combined runner still failed the original Torch bootstrap-identity test and
remains recorded as exit one. Native vmap legitimately installs its dispatch
wrappers lazily. Running that before the bootstrap identity assertion gives a
deterministic red; moving all nine original assertions into the existing fresh
mode probe gives green while the parent keeps its working vmap wrappers.
Only that Torch-owned test file changed. The maintained
`tools/run_test_suite.py --tier smoke --backend cpu --session torch` then passed
as `smoke-final-torch-r8`, retaining default four workers and strict execution
accounting. Official selection proves the changed file is outside all 636
Native-selected files and inside the 417 Torch-selected files. Between those two test runs, all other
3,066 source-map entries remained unchanged, including symbolic links and all
shared runtime and producer files. The one non-strict native XPASS also appears in earlier runs;
it is retained as an observation rather than credited as a fixed defect.

The current pinned text-Trainer gate also passes all sixteen methods on each
device, with three actual optimizer updates per runtime/method and zero
fallback. These checks use the original tiny random FP32 Qwen3 protocol and
cover sixteen of eighteen exported Trainers; `AlignPropTrainer` and
`DDPOTrainer` remain outside the text-only scope. Full-checkpoint FP16/BF16
acceptance is established separately for the SFT protocol above. CPU retains 3,700 observations and CUDA 3,697; each checks all 1,236
gradient and 1,236 parameter fields. PPO contributes three generations and
375 passing fields on each device. The maintained 128-sample mask stress
executes 448 actual asynchronous batches and passes both source guards.
These gates share all 1,279 runtime and 33 producer hashes with `formal-r7`.

Unversioned final receipts are under
`$JITTOR_LAB_ROOT/_state/trl-completion-20261026/`: numerical audit
`numerical/formal-r7/final-numerical-audit.json` (SHA256
`3431f26464d08adf303fd4235dde3696bef948cfbe16e52ae1c540fbd832fd00`),
`indexing/final-r7-cpu-cuda-audit.json`, and
`smoke/final-acceptance-source-binding.json` (SHA256
`e03c083222a8b390f2d99efc0567340ade18a45b4319be6a35aa2f80fd00ee37`).
The complete changed-Python differential Ruff check covers 118 files without
adding an issue to the preserved initial working tree or tracked baseline.
Repository layout and documentation governance checks pass.

A final directory-selection check exposed one controller-only accounting
boundary: pytest correctly ignored the native-owned ACL recorder in Torch
mode, while the execution ledger incorrectly retained it as an empty file.
The ledger now follows the same collection rule for directory walks and still
accounts for explicit file/node selections. True empty tests and unexplained
skips remain failures. All 62 accounting/contract controls pass, including five
new mode/explicit-selection cases. The original complete `tests/structure`
entry then passes with 1,528 passed, four explained skips and exit zero.

These two later test-side files do not belong to any measured numerical or
throughput producer/runtime map. The official native/Torch smoke invocations
already explicitly exclude the opposite mode: their 636/417 selected file sets
remain exactly unchanged. Their earlier 9,342-pass counts are retained at their
actual source boundaries; the later focused controls and full structure run are
recorded separately. Unversioned receipts are
`smoke/late-mode-ledger-scope-proof.json`, `mode-collection-ledger-green-r1.json`
and `precommit-structure-final-r2.json`. Earlier timeout-only synthetic attempts
and the strict structure ledger failure retain their failed statuses.

## Final throughput

The unchanged maintained `trl_throughput_probe.py` completes six priming runs
and all eighteen formal runs on one allocated RTX 2080 Ti (SM75, 250 W;
CUDA 12.8, driver 595.71.05). Each precision/runtime has three independent
fresh-process repetitions, with paired order alternated by the original driver.
The full Qwen3-0.6B base stays FP32 and the same rank-8 q/v LoRA adapters train.
Each run uses batch one, accumulation two, maximum length 128 and a constant
`1e-4` schedule: five warmup plus twenty timed optimizer updates. Every timed
interval contains 40 sequences, 3,352 nonpadding tokens and 3,312 causal targets.
This differs from the ten-step, length-32 numerical protocol above.

| Precision | Torch tokens/s, median [min, max] | Jittor tokens/s, median [min, max] | Jittor / Torch |
| --- | ---: | ---: | ---: |
| FP32 | 567.8 [557.8, 570.6] | 361.9 [358.4, 369.1] | 0.637 |
| FP16 | 451.9 [416.6, 482.6] | 302.2 [296.0, 304.7] | 0.669 |
| BF16 | 484.9 [458.5, 489.7] | 265.4 [260.0, 269.7] | 0.547 |

Jittor is slower than independent Torch in all three measured profiles. All
18 original acceptance rows pass: 25 actual updates, all 112 finite parameter
states with counters 25, 423 paired initial tensor/buffer hashes, zero fallback
and no newly compiled Jittor shared objects in the timed intervals. The four
outer source guards bind 1,340 runtime/producer files and the complete dynamic
1,271-owner inventory; compiled-core identity stays fixed within each runtime.

The synchronized Trainer-update interval excludes the initial CUDA-prefetched
accumulation group and lookahead; later in-loop loading/transfers and stock
Trainer metrics remain included. These rates do not establish complete
end-to-end input latency, allocator-memory superiority, model quality or native
BF16 hardware acceleration. Post-timing single-prompt hashes are stable within
each runtime's three repeats; FP32/FP16 hashes differ across runtimes and BF16
matches. All argmaxes are 220. These limited diagnostics do not grant strict
numerical acceptance to this separate 25-step performance profile; the complete
byte-exact result remains the original ten-step protocol above.

Unversioned raw rows, commands, hardware records, source maps and independently
recomputed summary are under
`$JITTOR_LAB_ROOT/_state/trl-completion-20261026/throughput-final-r7/`.
`final-throughput-audit.json` has SHA256
`7ccaad9b1a572cf33165f78a5ec584cae49e9deaee30f8b84695c7784001a460`;
all workers, stage drivers and the audit exit zero before subsequent checks or
commits. The runbook's full-model numerical and throughput sections reproduce
the two distinct protocols with external artifacts and exclusive caches.
