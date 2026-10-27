# TRL 0.21.0 selected Trainer coverage investigation

- Status: Completed bounded CPU/CUDA numerical investigation. No general-library or nightly admission is claimed.
- Date: 2026-10-22
- Baseline commit: `b9517ff138d424c4d7d6589795f95c8a1de9f56e`, on the `2.0-refactor` development line; each numerical run must additionally bind its actual working-tree changes and probe hashes.
- Owner: Jittor Torch compatibility maintainers
- Review when: The pinned TRL source, selected methods, model configuration, comparison protocol, or Jittor frontend changes.

## Version and scope

This investigation targets **TRL 0.21.0**, official upstream commit
`46d09bd2408f17605409fb3ee8ba12705add7faa`. It does not establish compatibility
with the latest TRL release. The [current TRL documentation](https://huggingface.co/docs/trl/index)
has a different Trainer organization, including a Distillation entry; that
current taxonomy must not be projected onto this older source snapshot. In
particular, the pinned public distillation class here is `GKDTrainer`, and the
pinned exports include PPO, ORPO, and CPO even though their location/status in
newer documentation differs.

[trl_inventory.py](../../agent/skills/torch-compat-repo-runbook/scripts/trl_inventory.py)
statically reads both upstream `_import_structure` declarations, including
conditional extensions. It does not import TRL, Torch, or optional dependencies.
It checks the official remote, clean checkout, full revision, packaging version,
actual class definitions, and source-file hashes.

The resulting **complete denominator is 18 public Trainer exports**. This round
selects 11 of those 18 for investigation. **Selection alone is not numerical
support.** A method contributes to numerical coverage only after
the applicable independent-runtime evidence is accepted. Conditional DDPO remains
in the denominator even when diffusers is absent from the evaluation environment.

| Method | CPU: ten updates | CUDA: ten updates | Final output max abs, CPU / CUDA | Next-token top-1, CPU / CUDA |
| --- | --- | --- | --- | --- |
| SFT | Strict pass | Strict pass | `8.941e-08` / `8.941e-08` | 10/10 / 10/10 |
| DPO | Strict pass | Strict pass | `1.192e-07` / `8.941e-08` | 10/10 / 10/10 |
| KTO | Strict pass | Strict pass | `8.941e-08` / `1.192e-07` | 10/10 / 10/10 |
| ORPO | Strict pass | Strict pass | `1.788e-07` / `1.192e-07` | 10/10 / 10/10 |
| CPO | Strict pass | Strict pass | `1.192e-07` / `8.941e-08` | 10/10 / 10/10 |
| Reward | Strict pass | Strict pass | `2.235e-08` / `2.980e-08` | Scalar reward / Scalar reward |
| GRPO | Strict pass | Strict pass | `1.192e-07` / `8.941e-08` | 10/10 / 10/10 |
| PPO | Strict pass | Strict pass | `9.686e-08` / `8.941e-08` | 10/10 / 10/10 |
| RLOO | Strict pass | Strict pass | `1.192e-07` / `8.941e-08` | 10/10 / 10/10 |
| GKD | Strict pass | Strict pass | `8.941e-08` / `1.192e-07` | 10/10 / 10/10 |
| Online DPO | Strict pass | Strict pass | `1.192e-07` / `7.451e-08` | 10/10 / 10/10 |

Final output is the next-token vocabulary vector for causal policies and the
scalar logit for Reward. All earlier forward/gradient/update observations must
also pass; this column is not a substitute for whole-run acceptance.

The seven remaining exports stay visible in the denominator:

| Export outside this round | Reason |
| --- | --- |
| AlignPropTrainer | Diffusion/image training; explicitly deprecated in the pinned class source |
| DDPOTrainer | Conditional diffusion/image training; explicitly deprecated in the pinned class source |
| BCOTrainer | An additional binary-classifier preference objective outside the selected subset |
| IterativeSFTTrainer | An older iterative SFT interface distinct from the selected SFTTrainer path |
| NashMDTrainer | An additional online preference objective outside the selected subset |
| PRMTrainer | Process-level reward modeling, distinct from the selected sequence reward task |
| XPOTrainer | An additional exploration-based online preference objective outside the selected subset |

The last five exclusions are scope decisions; this report does not label them
deprecated without an upstream declaration.

## Tiny complete-model experiments and existing full-checkpoint evidence

The new breadth matrix uses small, randomly initialized Qwen3 models with their
complete forward/backward architecture and official Trainer objective paths.
Reduced model dimensions and synthetic token sequences make many methods
practical to compare. These are **not the pretrained Qwen3-0.6B checkpoint** and
do not measure language quality, full-model memory, or deployment throughput.
The current probe configuration has two decoder layers, hidden size 32,
intermediate size 64, four query heads, two key/value heads, head dimension 8,
and a vocabulary of 256 tokens. It uses FP32 and eager attention.
The causal policy has 26,816 trainable parameters in 24 unique tensors. The
sequence reward model has 26,848; PPO trains both policy and value model,
53,664 parameters total. All use SGD at `1e-4`, without LoRA or frozen policy
layers. The same downstream stack as the full-checkpoint reports is used:
PyTorch 2.8.0+cu128, Transformers 4.56.2, Accelerate 1.10.1, datasets 3.6.0,
NumPy 1.26.4, and four CPU threads. CUDA uses one RTX 2080 Ti, FP32/eager,
TF32 disabled; no AMP or distributed training.

Offline cases retain official loss implementations and explicitly control the
sample order. KTO needs a real batch larger than one; its batch of two
means ten optimizer steps consume twenty examples, rather than ten. Reward
training uses an actual sequence classification head. Online cases use the
official generation/scoring/training paths with a controlled token-sampling
replay where needed for independent-runtime comparisons. Such replay tests
training on matched rollouts; it does not establish identical independent RNG
behavior or unconstrained generation. Any additional ordering control must be
recorded in the case metadata.

Every policy, reference, teacher, reward, and value model used by a case needs
explicitly synchronized initial state. Numerical claims require the actual
inputs/outputs, objective observations, all requested trainable gradients, all
updated parameters, device observations, and zero-fallback evidence. PPO's value
model is trainable and must be included alongside its policy. Frozen roles must
remain unchanged. Import success, a completed loop, or a falling loss alone is
not numerical admission.

The shared comparison currently keeps the existing elementwise error budget:
`abs(candidate - reference) <= 1e-5 + 1e-4 * scale`, where
`scale = max(abs(reference_element), max_abs(reference_field) * 1e-6, 1e-30)`.
Fields marked exact, including integer training inputs and replayed tokens,
must match exactly. Reported relative L2 error is an additional diagnostic and
does not override a failed elementwise comparison.

The earlier full-checkpoint experiments remain separate:

- [Qwen3-0.6B SFT investigation](2026-10-21-trl-qwen3-sft.md).
- [Qwen3-0.6B DPO/GRPO investigation](2026-10-21-trl-dpo-grpo-qwen3.md).

Those reports distinguish completed training from strict numerical acceptance.
In particular, the full-checkpoint CPU comparisons and the earlier DPO CUDA
experiment must not be relabeled numerical passes merely because a new tiny
case passes. Conversely, a tiny failure does not erase the recorded fact that a
separate full-checkpoint execution completed. Coverage and experiment scale are
separate axes.

## Source-bound inventory evidence

The inventory is reproducible without a tensor runtime:

```bash
python agent/skills/torch-compat-repo-runbook/scripts/trl_inventory.py \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --revision 46d09bd2408f17605409fb3ee8ba12705add7faa \
  --output "$JITTOR_LAB_ROOT/_state/trl-mainstream/inventory.json"
```

Raw inventory: `$JITTOR_LAB_ROOT/_state/trl-mainstream/inventory.json`
(**unversioned**), SHA256
`e3e901acedd00a688771d362abe80fc3b773a4001213e389db248c41d110ac46`.
It records the class-source SHA256 for all 18 Trainers and declares
`coverage_claim: null`.

| Source | SHA256 |
| --- | --- |
| `trl/__init__.py` | `1b43654910f88ff95b84fa7f069e7e3f11e8286a9e902c28899e81f394fea9c9` |
| `trl/trainer/__init__.py` | `951896ac1209eced7c1db21a8418c7cbcbab4b137793e543627f0e23b20d3d5d` |
| `setup.cfg` | `2d5e1534ff33707931162779b847b445ffed614836a60d1eaeac51f98f16366f` |

Pinned packaging declares Python `>=3.9`, Accelerate `>=1.4.0`, Datasets `>=3.0.0`,
and Transformers `>=4.55.0`. These are upstream minimum requirements, not the
versions used by a numerical experiment. The inventory also preserves all
declared optional-dependency extras; actual environment versions and installed
source bindings belong to each run's provenance.

Offline inventory checks: **11 passed, zero skipped**. They verify conditional
exports, automatic denominator growth, class resolution, and rejection of
dynamic/ambiguous declarations, wrong revisions/remotes, dirty sources, and
version mismatches. Ruff passed for the new inventory tool and its tests. These
checks validate inventory construction, not Trainer behavior.

## Candidate source identity

The current tracked working-tree patch was archived at `2026-10-22T07:13:39Z`
against baseline `b9517ff138d424c4d7d6589795f95c8a1de9f56e`:

- Patch: `$JITTOR_LAB_ROOT/_state/trl-mainstream/source-patch/20261022T071339Z.patch` (unversioned).
- Receipt: `$JITTOR_LAB_ROOT/_state/trl-mainstream/source-patch/20261022T071339Z.json` (unversioned).
- Patch SHA256: `cf1210b43453bd9a08114cd1569f7ca17b6eb8a3197b3bd3f14c645615135ae9`.
- `compat/torch/installers/tensor/method_api.py` SHA256: `327eebcc0fbaffaea1a71375759ae6ad6c2997c84ef6499706cd5e16c49556c6`.

The receipt includes hashes of the tracked changed files and the observed Git
status. New, untracked probe files are bound separately by each run's
`tool_sources` hashes. This snapshot identifies the source at capture time; it
must not automatically be assigned to earlier or later measurements. Further
implementation changes need a new receipt and a review of affected scenarios
before their results can be combined with this matrix.

## Numerical matrix results

All **22/22 selected method/device cells** complete ten updates and pass
the fixed strict comparison budget, with zero fallback. This is **11/18 public
Trainers at the pinned SHA on each device**, only at the stated synthetic
configuration. The seven unselected Trainers and other configurations remain
unverified; these counts do not imply broad checkpoint or objective coverage.

The accepted pairs are audited separately from exploratory and failed runs.
Each completed pair must have ten updates, zero fallback, identical model-role
initial tensors, identical protocol/source/parameter inventories, all parameter
gradients and all updated parameters at every step. The external audit also
rehashes every saved observation and initial model artifact. No speed or memory
ratio is inferred from these instrumented, first-compilation runs.

The exploratory runner may finish with exit code zero after recording a
numerical mismatch so that all observations can be collected. Matrix admission
therefore reads `status`, completed optimizer steps, device/fallback evidence,
and the separate `numerical_passed` result. Process completion alone is never
interpreted as a numerical pass.


## PPO index compatibility and environment corrections

The first online candidate attempts lacked Jinja2. The candidate environment
was completed with the oracle's exact Jinja2 3.1.6 and MarkupSafe 3.0.3 versions,
using dependency-free installation; it still imports the Jittor Torch shim.
`online-dependency-receipt.json` records both environments. These failed imports
are excluded from training/numerical results and are not framework failures.

PPO then reproduced a real frontend indexing defect at upstream
`ppo_trainer.py:516`: `rewards[[actual_start, actual_end]] += scores`.
The short list contains one Tensor index per axis. Jittor originally treated
it as one stacked index, producing `[2,2,4]` instead of `[2]`. The
[PyTorch 2.11 indexing implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/python_variable_indexing.cpp)
still recognizes this non-tuple sequence as multiple axes. Its warning about a
future 2.9 change is insufficient evidence that the behavior actually changed.

The Torch frontend now normalizes short non-array sequences containing Tensor,
sequence, slice, None or ellipsis entries to an outer tuple, with a deprecation
warning. Flat integer lists, empty lists and sequences of length at least 32
retain their single-index interpretation. The same normalization is used for
reads and writes; CPU index tensors are copied when needed for accelerator data
without moving the caller's index. Native Jittor semantics and the official TRL
training implementation remain unchanged. This fixes the read and write sides
of augmented assignment together.

The independent Torch CPU regression passes nine cases; the unfixed Jittor
CPU regression failed six and passed three. After correction, all nine Jittor
CPU cases pass, covering actual values, gradient scatter, assignment, augmented
assignment, index identity/residency, sequence forms and the 32-item boundary.
The existing ten-case CPU advanced-index regression also passes.

A second CUDA check caught `Tensor.tolist()` migrating the original Tensor's
physical allocation to CPU while its device tag remained CUDA. Upstream GRPO
calls tokenizer decoding before generation, which exposed this mismatch.
The new matrix/scalar/view regression failed all three CUDA cases before the
fix. The frontend now reads a separate CPU copy; all three cases then pass,
including the original Tensor's device, view storage and subsequent gradient.
No fallback assertion was disabled and no extra transfer was added to TRL.

The final candidate matrix is rerun on one frozen implementation after both
fixes, rather than mixing earlier attempts with later source. The patch and its
SHA256 are recorded in `final-implementation.patch` and
`final-implementation-source.json`. Source-bound oracle artifacts are reused:
the observation tools, Trainer mathematics and initial-state protocol did not
change. Earlier candidate attempts remain diagnostic evidence only.

## Evidence layout and bounds

The [runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md) contains
standalone inventory and paired experiment commands. Raw results are outside
the repository in `$JITTOR_LAB_ROOT/_state/trl-mainstream/` (unversioned):

- Independent oracles: `<method>-torch-<device>-10/trace`.
- Accepted final candidates: `<method>-jittor-<device>-10-final/trace`.
- Earlier unsuffixed, retry, smoke and diagnostic directories are excluded
  from the final accepted matrix.
- `audited-matrix.json`: explicit selected trace mapping, manifest hashes,
  category errors, final-output errors and next-token decisions.
- `formal-source.json`, environment receipts and the two source-patch receipts.

Method names in paths are lowercase, with `online_dpo` for Online DPO. The
external summary script under `$JITTOR_LAB_ROOT/trl-qwen3-sft/` is unversioned;
the maintained probes, tests and reproduction commands are in the repository.

Each new method uses ten distinct synthetic inputs. KTO consumes twenty example
visits. PPO uses five rollout batches of two prompts and two minibatches per
rollout, giving ten optimizer updates and ten generated sequences. RLOO uses
those ten prompts with two completions each, twenty sequences and ten updates.
GRPO uses two completions per prompt, a synthetic token-based reward and
canonical ordering within each two-answer group. GKD uses the on-policy branch
with `lmbda=1`; its teacher is an independent fixed random model.

Final evaluation uses ten fixed synthetic prompts and each runtime's own trained
policy. Causal methods compare full-vocabulary logits and next-token vectors;
Reward compares scalar logits. Matching one next token does not establish
matching generated text. Only sampled token choices are replayed in online
training: native forward, local probabilities and generation scores are still
computed in each runtime. No claim follows about unconstrained sampling,
semantic reward quality, pretrained-model convergence, long sequences, every
objective variant, LoRA/AdamW/AMP, distributed execution or current TRL versions.
These manual experiments are not nightly or general-library admission.

Final implementation SHA256: `a18f48479e7ec5089d6ee6d3d0b602616dc06123b4707d2651194d8ccf9c7bbc` for
`compat/torch/installers/tensor/method_api.py`. The full-checkpoint report
retains the earlier implementation identities it actually measured.

### Largest observed training errors

| Method | Max parameter-gradient error, CPU / CUDA | Max updated-parameter error, CPU / CUDA |
| --- | --- | --- |
| SFT | `2.459e-07` / `1.788e-07` | `3.725e-09` / `3.725e-09` |
| DPO | `8.941e-08` / `5.960e-08` | `1.863e-09` / `3.725e-09` |
| KTO | `1.490e-08` / `1.118e-08` | `3.725e-09` / `4.657e-10` |
| ORPO | `4.768e-07` / `2.384e-07` | `3.725e-09` / `3.725e-09` |
| CPO | `7.153e-07` / `3.576e-07` | `3.725e-09` / `1.192e-07` |
| Reward | `5.960e-07` / `3.576e-07` | `3.725e-09` / `3.725e-09` |
| GRPO | `1.490e-07` / `1.490e-07` | `3.725e-09` / `1.863e-09` |
| PPO | `1.013e-06` / `8.345e-07` | `3.725e-09` / `3.725e-09` |
| RLOO | `2.801e-06` / `9.537e-07` | `3.725e-09` / `3.725e-09` |
| GKD | `6.272e-09` / `6.548e-09` | `2.910e-11` / `5.821e-11` |
| Online DPO | `8.941e-08` / `8.941e-08` | `1.863e-09` / `9.313e-10` |


## Final repository validation

- Native core: 98 passed, 41 skipped, one expected failure.
- Torch compatibility core: 224 passed, one skipped.
- Offline observation/inventory/builder controls: 83 passed.
- Repository structure: 1,516 passed, four skipped.
- New multi-axis sequence regression: nine passed on each of CPU/CUDA in both
  genuine Torch and Jittor.
- New `tolist` residency regression: three passed on each device in both runtimes.
- Jittor CPU-index regression: ten CPU and twelve CUDA cases passed.
- New tools/tests pass Ruff 0.15.22; the changed implementation introduces no
  new diagnostics over its 13 pre-existing import diagnostics.
- Layout, generated manifest and diff whitespace checks pass.

The first full structure check rejected a test helper that imported a module
at collection time. It was moved into a fixture; both its functional tests and
the complete structure suite were then rerun successfully. The earlier failure
is retained under `checks-before-final/`, not counted as final success. Final
logs and exit receipts are under `$JITTOR_LAB_ROOT/_state/trl-mainstream/`;
index and residency test logs are under `$JITTOR_LAB_ROOT/_state/trl-rl-qwen3/`.
The matrix audit revalidated every initial artifact and every saved observation
payload hash for all 22 selected pairs, including their actual gradient and
updated-parameter inventories. All 200 next-token choices (100 per device,
excluding scalar Reward outputs) match the independent oracle.
