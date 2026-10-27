# TRL DPO and GRPO: ten-example full Qwen3-0.6B experiments

- Status: Complete bounded measurements on CPU and CUDA. DPO and GRPO each complete ten updates on both runtimes but fail strict numerical acceptance. Trained-weight logits are complete: next-token top-1 agrees on all ten prompts per method/device, but strict element budgets fail. No nightly or library-wide admission is claimed.
- Reviewed: 2026-10-22.
- Owner: Torch compatibility and downstream verification maintainers.
- Source: upstream `2.0-refactor` commit `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`, local SFT compatibility commit `b9517ff138d424c4d7d6589795f95c8a1de9f56e`, plus frozen observation tools and the CPU advanced-index compatibility fix identified below.
- Review when: checkpoint, downstream versions, Trainer code, rewards, generation protocol, reduction/gradient implementations, precision, probe source or error budgets change.

## Scope and fixed conditions

This extends the [three-step SFT experiment](2026-10-21-trl-qwen3-sft.md)
to official TRL `DPOTrainer` and `GRPOTrainer`. Both use the complete pretrained
Qwen3-0.6B model: **28 layers, 596,049,920 trainable parameters, 310 unique
parameter tensors**. Embeddings and the LM head remain tied. No layers are
removed or frozen, and no LoRA or handwritten optimizer replaces upstream
training mathematics.

| Item | Fixed choice |
| --- | --- |
| Checkpoint | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| TRL | 0.21.0, official source `46d09bd2408f17605409fb3ee8ba12705add7faa` |
| Transformers / Accelerate / datasets | 4.56.2 / 1.10.1 / 3.6.0 |
| Independent oracle | Genuine PyTorch 2.8.0+cu128, separate interpreter |
| Candidate | Current Jittor core and Torch compatibility shim, without genuine Torch |
| Precision / attention | FP32 / eager, TF32 off, four CPU threads |
| Optimizer | SGD, learning rate `1e-4`, constant schedule, no momentum, decay or clipping |
| Volume | Ten updates per method, ten distinct prompts, each pair/group used once |
| Other conditions | No AMP, gradient checkpointing, workers, pinning, distributed training or vLLM |
| CPU host | Dual Xeon Platinum 8255C, 96 logical CPUs; each model process uses four threads |
| CUDA target | Single RTX 2080 Ti 11 GB, SM75, CUDA toolkit 12.8.93, driver 595.71.05 |

Both sides load identical initial parameter bytes and shared oracle-created
nonpersistent buffers, including RoPE frequencies. The buffer artifact SHA256
is `5dd3d11408f2d167cb5d69044fd20c9742b16fdbfccb3ff8a3a9e93fc1ee2b91`.
The harness checks complete initial policy/reference hashes, dependency versions,
installed TRL source bytes, probe hashes, observation inventories, input types
and array hashes. Jittor forbids fallback and verifies physical placement and
native dispatch of initial parameters, every parameter gradient, final
parameters and retained reference state.

The subclasses retain upstream training mathematics. DPO specifies a sequential
sampler and observes the original loss/metrics. GRPO observes original reward,
log-probability and loss routines, and consistently reorders all batch-aligned
inputs after the upstream shuffle. These are explicit experiment controls, not
a new general TRL adapter.

## CUDA advanced-index compatibility fix

The first Jittor CUDA GRPO attempt stopped before an optimizer update in
upstream `truncate_with_protected_tokens`. That helper constructs a CPU boolean
mask, then uses it to select CUDA token IDs. Genuine PyTorch permits CPU
advanced indices for accelerator data; Jittor's native operators deliberately
require matching device placement. This was a missing frontend conversion,
not an incorrectly executed training update or a backend fallback.

`_torch_getitem` now explicitly copies permitted CPU boolean/integer indices to
the indexed tensor's device, including tuple indices and converted NumPy
indices. It leaves the original index on CPU. Indices from a different GPU are
not moved, and native device checks remain unchanged. No TRL source or training
formula was modified. The failed trace remains under
`grpo-jittor-cuda-10/`; the corrected full run uses a new
`grpo-jittor-cuda-10-r2/` directory rather than overwriting it. An intermediate
`grpo-jittor-cuda-10-fixed/` attempt was interrupted after core regression
exposed native Var device strings; neither incomplete attempt is accepted.

`compat/tests/torch/test_cpu_advanced_index.py` passes ten CPU and twelve CUDA
cases on Jittor. Independent PyTorch passes seven CPU and nine CUDA cases,
with three native-Var-only cases skipped on each device. Coverage includes
boolean/int32/int64 indices, empty selection, tuple mixtures, duplicate-index
gradients, unchanged source-index placement, and rejection of CUDA indices
for CPU data or a different GPU. The shared device helpers accept both native
Var string devices and Torch Tensor device objects. Native legacy indices
retain their original device/location/placement/dispatch state. The previously
failing core NLL forward/backward test also passes on CPU and CUDA. Jittor
positive cases check physical placement and zero fallback. The final fix is
frozen at `method_api.py` SHA256
`327eebcc0fbaffaea1a71375759ae6ad6c2997c84ef6499706cd5e16c49556c6`.

CPU runs and DPO training completed before this fix; they did not exercise the
previously failing CPU-index/accelerator-data path. Corrected GRPO CUDA and
Jittor CUDA post-training logits use the fix. These source scopes are recorded
separately instead of relabeling older measurements as having tested it.

## DPO protocol

Ten short arithmetic/factual questions each have a chosen and rejected answer,
such as `2 + 3 -> 5` versus `8`. Full records are in `trl_dpo_case.py` and each
manifest. One pair forms each update; the forward pass includes both answers.
Maximum prompt/completion/total lengths are 48/16/64.

The loss is upstream sigmoid DPO, beta `0.1`, no label smoothing, with an actual
reference policy. Official `precompute_ref_log_probs=True` computes fixed
reference log probabilities from the initial policy without retaining a second
model. Both train and evaluation dataloaders are prepared before any update,
preventing evaluation from using the final policy as its reference.

Before/after evaluation uses the **same ten training pairs**, not a held-out
dataset. DPO `rewards/accuracies` compares chosen/rejected implicit rewards;
it is not factual answer accuracy. At initial policy/reference equality, tiny
rounding differences can change its strict greater-than decision around zero.

## GRPO protocol and generation boundary

Ten different questions each generate two completions: **20 completions and ten
updates**. Batch size and `num_generations` are two; gradient accumulation,
`steps_per_generation` and `num_iterations` are one. Maximum prompt/completion
lengths are 64/16. Loss is explicitly `grpo` rather than this version's default
`bnpo`; beta is `0.04`, with a separate unchanged initial reference model and
reward scaling enabled.

The dense diagnostic reward is `0.8 * character_sequence_similarity +
0.2 / (1 + abs(answer_length - target_length))`, after stripping whitespace and
case-folding both strings. Similarity uses Python `SequenceMatcher` with
`autojunk=False`. This is a character/length test reward, not a validated
semantic correctness judge. Actual answers/rewards are retained, equal-reward
groups are counted, and sampling is never retried to manufacture advantages.
Both completed CPU runs have **10/10 groups with distinct rewards**.
Both completed CUDA runs instead have **9/10** distinct-reward groups. For prompt `What is 10 divided by 2?`
(zero-based step 3), both sampled completions are exactly
`5\nAnswer: 5\nQuestion: What is 10 divided by`, with equal reward
`0.04101479915433404`. That group's normalized reward advantages are zero.
All ten CUDA steps in both runtimes nevertheless have nonzero parameter
gradients; step 3 has gradient L2 `0.55248381` in PyTorch and `0.55247743` in
Jittor, consistent with the enabled KL term still providing a training signal. Nonzero gradients must not be reported as ten nonzero
reward-variance groups.

PyTorch performs real sampling at each current policy state, with temperature
1, top-p 1, top-k 0 and cache disabled. `use_model_defaults=False` prevents
Transformers from replacing global defaults with checkpoint-specific defaults
such as temperature `0.6` and top-p `0.95`. Every call saves its effective
generation configuration, prompt IDs, attention mask and generated sequence in
`rollouts.json` format 2.

Jittor verifies each prompt/configuration, then supplies the **identical oracle
token sequence** at the model's `generate` boundary. Official completion
masking, reference log probabilities, rewards, normalization, KL, policy loss
and backpropagation still execute. Batch rows are sorted by prompt/completion
token IDs with masks and advantages moving together. This isolates training
numerics from different RNGs and discrete samples; it does **not** establish
independent online GRPO rollout parity.

With one iteration per rollout, upstream uses detached current log probabilities
as the old-policy baseline. Importance ratios equal one at the update, so this
does not demonstrate an active clipping boundary or multi-update rollout reuse.

After training, the official helper re-scores fixed completions. The report
compares each group's mean completion log probabilities immediately before
**that group's own update** with the final policy after ten updates. Before
values come from different policy states. The high-reward minus low-reward
log-probability gap is a training-set diagnostic, not a common-initial-policy
evaluation or evidence of better rewards under fresh sampling.

A separate external native-generation smoke was completed only on PyTorch CPU
using initial pretrained weights. It supplies no Jittor generation evidence and
does not compare trained-model outputs. The relevant output comparison for this
experiment is the deterministic trained-weight logit probe below.

## Error definitions and current results

Every gradient of all 310 parameters is observed at each update: **3,100
gradient tensors per run**. All 310 final parameter tensors are compared,
alongside losses, fixed inputs and algorithm-specific observations. This harness
does not collect full model logits, input/embedding gradients or a complete
updated-parameter snapshot after every step.

Discrete inputs require exact agreement. The unchanged floating-point budget is:

```text
abs(candidate - reference) <= 1e-5 + 1e-4 * max(abs(reference), max_abs(reference_field) * 1e-6, 1e-30)
```

Global relative L2 is `sqrt(sum(error_squared) / sum(reference_squared))` over
the stated category. Final weights also use the actual oracle update norm
(`final - initial`) as denominator, avoiding misleadingly tiny errors relative
to large pretrained weights. Category L2 does not override failed element
budgets. Integer token IDs must not dominate an overall numerical headline.

| Method / device | Updates, PyTorch / Jittor | Strict result | Fallback |
| --- | --- | --- | --- |
| DPO CPU | 10 / 10 | Fail: 49 of 3,920 fields; 440 elements | 0 / 0 |
| GRPO CPU | 10 / 10 | Fail: 45 of 3,570 fields; 592 elements, all gradients | 0 / 0 |
| DPO CUDA | 10 / 10 | Fail: 67 of 3,920 fields; 537 elements | 0 / 0 |
| GRPO CUDA | 10 / 10 | Fail: 37 of 3,570 fields; 445 elements | 0 / 0 |

| DPO CPU observation | Maximum absolute difference | Global relative L2 | Strict failures |
| --- | ---: | ---: | ---: |
| Training losses | `2.14577e-6` | `1.66902e-6` | 0 / 10 fields |
| All gradients | `1.47820e-4` | `1.64327e-5` | 47 / 3,100 fields; 438 elements |
| Final parameters | `1.19209e-7` | `1.97650e-9` | 0 / 310 fields |
| Evaluation losses | `1.66893e-6` | `1.19359e-6` | 0 / 20 fields |

Final DPO CPU parameter error relative to the **actual update** is `1.24693e-4`
(approximately `0.01247%`). Two further failures are an initial reward-ranking
indicator (`0` versus `1` for one pair) and a rejected-reward scalar (absolute
difference `1.64032e-5`). The indicator explains overall maximum difference
`1`; that number is not a weight or gradient error.

| DPO CPU, same training set | PyTorch before -> after | Jittor before -> after |
| --- | --- | --- |
| Mean DPO loss | `0.69314718 -> 0.46264046` | `0.69314736 -> 0.46264076` |
| Mean implicit reward margin | `0 -> 0.58364785` | `-3.8147e-7 -> 0.58364737` |
| Mean chosen sequence log probability | `-22.308813 -> -19.525846` | `-22.308802 -> -19.525854` |
| Mean rejected sequence log probability | `-26.184834 -> -29.238348` | `-26.184813 -> -29.238342` |

| GRPO CPU observation | Maximum absolute difference | Global relative L2 | Strict failures |
| --- | ---: | ---: | ---: |
| Training losses | `2.98023e-7` | `3.06615e-5` | 0 / 10 fields |
| All gradients | `1.20163e-4` | `1.80722e-5` | 45 / 3,100 fields; 592 elements |
| Final parameters | `2.38419e-7` | `2.38716e-9` | 0 / 310 fields |
| Group advantages | `5.96046e-8` | `2.71138e-8` | 0 / 10 fields |
| Log probabilities and entropies | `1.56164e-4` | `7.03207e-6` | 0 / 60 fields |

GRPO CPU rewards and all recorded discrete inputs agree exactly. Its 45 failing
fields are entirely parameter gradients. Final parameter error relative to the
**actual update** is `1.98972e-4` (approximately `0.01990%`); the oracle update
L2 norm is `0.01728637`, and final cross-runtime weight-error L2 is
`3.43950e-6`. The largest absolute difference overall is an entropy observation
that still passes its magnitude-dependent budget; it does not identify the
worst failing gradient.

| GRPO CPU, retained completions | PyTorch | Jittor |
| --- | ---: | ---: |
| Distinct-reward groups | 10 / 10 | 10 / 10 |
| Mean high-minus-low log-probability gap before each group's own update | `0.37442311` | `0.37442220` |
| Mean gap under final policy | `1.13913912` | `1.13913558` |
| Mean gap improvement | `0.76471601` | `0.76471337` |
| Groups whose gap improves | 10 / 10 | 10 / 10 |

Both runtimes increase the relative log probability of the higher-reward fixed
completion in all ten groups. These are mean per-token log probabilities;
the before values use each group's current policy, not one shared initial
policy. No claim follows about newly generated answers or held-out quality.
The final mean gap differs across runtimes by approximately `3.54e-6`.

| DPO CUDA observation | Maximum absolute difference | Global relative L2 | Strict failures |
| --- | ---: | ---: | ---: |
| Training losses | `3.39746e-6` | `2.26048e-6` | 0 / 10 fields |
| All gradients | `1.92225e-4` | `1.96788e-5` | 66 / 3,100 fields; 536 elements |
| Final parameters | `2.38419e-7` | `2.01023e-9` | 0 / 310 fields |
| Evaluation losses | `1.07288e-6` | `8.08481e-7` | 0 / 20 fields |

DPO CUDA final parameter error relative to the actual update is `1.26821e-4`
(`0.01268%`). The remaining strict failure is one initial reward-ranking
indicator at a near-zero margin. Mean training-set evaluation loss changes from
`0.69314718` to `0.46263990` in PyTorch and to `0.46264043` in Jittor. The final
mean implicit reward margins are `0.58364952` and `0.58364803` respectively.

| GRPO CUDA observation | Maximum absolute difference | Global relative L2 | Strict failures |
| --- | ---: | ---: | ---: |
| Training losses | `2.77162e-6` | `3.15485e-5` | 0 / 10 fields |
| All gradients | `1.45435e-4` | `2.34119e-5` | 36 / 3,100 fields; 444 elements |
| Final parameters | `1.19209e-7` | `2.14203e-9` | 0 / 310 fields |
| Group advantages | `0` | `0` | 0 / 10 fields |
| Log probabilities and entropies | `1.25527e-4` | `4.75535e-6` | 1 / 60 fields; 1 element |

GRPO CUDA rewards, advantages and all recorded discrete inputs agree exactly.
Final parameter error relative to the **actual update** is `2.02833e-4`
(approximately `0.02028%`); the oracle update L2 norm is `0.01521601`, and
cross-runtime final weight-error L2 is `3.08631e-6`. In addition to the 36
failing gradient fields, one policy token log probability at zero-based step 9
fails its element budget: PyTorch `-0.25646210` versus Jittor `-0.25642395`,
absolute difference `3.81470e-5` against a `3.56462e-5` budget. The largest
entropy difference passes its magnitude-dependent budget.

| GRPO CUDA, retained completions | PyTorch | Jittor |
| --- | ---: | ---: |
| Distinct-reward groups | 9 / 10 | 9 / 10 |
| Mean high-minus-low log-probability gap before each group's own update | `0.37715866` | `0.37715866` |
| Mean gap under final policy | `0.84644516` | `0.84644844` |
| Mean gap improvement | `0.46928649` | `0.46928979` |
| Distinct-reward groups whose gap improves | 9 / 9 | 9 / 9 |

These means cover only the nine groups with distinct rewards. The equal-reward
group has no defined higher-reward completion and is excluded. All nine gaps
improve in both runtimes; the final mean gap differs by `3.28488e-6`. As for
CPU, these are fixed-completion diagnostics with different before-policy
states, not fresh-generation quality measurements.

## Post-training full-vocabulary and next-token logits

`trl_final_logits_probe.py` loads each runtime's **own trained final weights**
from its completed ten-update trace. It verifies the complete 310-parameter
inventory, each saved array's hash/shape/dtype, then copies every parameter to
the model and reads it back to verify the exact hash. Leaving some original
checkpoint parameters loaded cannot silently pass. Shared buffers and original
model/tokenizer artifacts are also checked against training provenance.

Each method uses its ten fixed training prompts, with exact token-ID agreement
between runtimes. In FP32/eager evaluation mode without sampling or cache, the
probe saves full logits **`[1, T, V]`** for all prompt positions and separately
the final position's next-token vector **`[V]`**, where `V=151936`. It reports
maximum absolute error, RMSE, relative L2, unchanged strict-budget failures,
top-1 token agreement and `KL(softmax(Torch) || softmax(Jittor))`, computed in
FP64 for diagnostics. Whole-prompt top-1 agreement and ten next-token decisions
have different denominators and must be reported separately.

These errors combine **training drift and inference arithmetic differences**:
the models intentionally use their own resulting weights. This is neither a
same-weight operator test nor random generation/quality evaluation. The probe
reuses saved weights and does not rerun or change the frozen training process.

With the same isolated offline CPU environment as training, run the independent
oracle first, then the candidate. `<oracle-training-trace>` and
`<candidate-training-trace>` are completed trace directories for the same method
and device; `--reference` points to the **oracle logit run**, not training:

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_final_logits_probe.py \
  --root "$PWD" --runtime torch --device cpu --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --training-trace <oracle-training-trace> \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/logits-torch-cpu"

<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/trl_final_logits_probe.py \
  --root "$PWD" --runtime jittor --device cpu --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --training-trace <candidate-training-trace> \
  --reference "$JITTOR_LAB_ROOT/_state/<new-run>/logits-torch-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/logits-jittor-cpu"
```

Repeat with distinct paths for DPO/GRPO, then matching CUDA training traces and
`--device cuda`. The [runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md)
gives explicit CPU environment prefixes and trace examples. A completed run
alone does not imply every element passes.

| Trained method / device | Prompt count | Max absolute logit error | Relative L2 | Top-1 agreement |
| --- | ---: | --- | --- | --- |
| DPO CPU | 10 | `7.91550e-5` | `2.99010e-6` | 10 / 10 |
| GRPO CPU | 10 | `1.41144e-4` | `5.94592e-6` | 10 / 10 |
| DPO CUDA | 10 | `9.34601e-5` | `3.74663e-6` | 10 / 10 |
| GRPO CUDA | 10 | `1.24931e-4` | `6.42362e-6` | 10 / 10 |

The table above is the **next-token vector** comparison. Each method/device pair compares
`10 * 151936 = 1,519,360` logits. All eight inference manifests are complete
with zero fallback; their recorded training-manifest hashes, all 310 loaded
parameter hashes, shared protocol/source identity and ten prompt-token hashes
were cross-checked. Each model therefore uses its own completed ten-update
training result. This documentation audit does not independently reload every
large weight payload; the inference probe verified those payloads and exact
parameter bytes when loading.

| Next-token vectors | Strict failed fields / elements | RMSE | Mean softmax KL, Torch to Jittor | Maximum KL |
| --- | --- | ---: | ---: | ---: |
| DPO CPU | 10 / 10 fields; 7,726 / 1,519,360 elements | `1.15114e-5` | `1.66576e-11` | `5.10271e-11` |
| GRPO CPU | 10 / 10 fields; 16,047 / 1,519,360 elements | `1.65195e-5` | `7.27330e-11` | `1.93427e-10` |
| DPO CUDA | 10 / 10 fields; 11,559 / 1,519,360 elements | `1.44239e-5` | `1.88125e-11` | `4.15651e-11` |
| GRPO CUDA | 10 / 10 fields; 16,378 / 1,519,360 elements | `1.85762e-5` | `7.66843e-11` | `3.15446e-10` |

| Logits at all prompt positions | Max absolute error | Relative L2 | Strict failed fields / elements | Top-1 agreement | Mean softmax KL |
| --- | ---: | ---: | --- | --- | ---: |
| DPO CPU: 101 positions, 15,345,536 logits | `1.26600e-4` | `2.54666e-6` | 10 / 10 fields; 25,620 elements | 101 / 101 | `1.47740e-11` |
| GRPO CPU: 139 positions, 21,119,104 logits | `2.86102e-4` | `3.74156e-6` | 10 / 10 fields; 91,871 elements | 139 / 139 | `4.32980e-11` |
| DPO CUDA: 101 positions, 15,345,536 logits | `1.21355e-4` | `2.44056e-6` | 10 / 10 fields; 22,837 elements | 101 / 101 | `9.07467e-12` |
| GRPO CUDA: 139 positions, 21,119,104 logits | `1.88828e-4` | `4.62019e-6` | 10 / 10 fields; 118,956 elements | 139 / 139 | `4.11897e-11` |

The distributions and maximum-logit choices are close, while the original
strict element budgets still fail; neither observation replaces the other.
Next-token agreement concerns one continuation token, which may be whitespace
or a numeric fragment. It does not establish identical full generated answers
or semantic correctness. CPU and CUDA numbers are separate within-device
comparisons; they do not establish identical CPU/CUDA training trajectories.

## Reproduction and evidence

Use the [full-checkpoint DPO/GRPO runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md)
and independent interpreters. Run the oracle first, matching devices and ten
updates. Reuse the shared buffer artifact from the SFT setup. Run CPU before
CUDA, serialize first compilation, and use separate private `JITTOR_HOME`
directories per device. The candidate uses `DISABLE_MULTIPROCESSING=1`,
`JT_USE_PARALLEL_OP_COMPILER=0`, `JT_CUDA_KERNEL_MATH=strict` and four
OpenMP/MKL/OpenBLAS threads.

Unversioned evidence is under `$JITTOR_LAB_ROOT/_state/trl-rl-qwen3/`:

- `dpo-torch-cpu-10/trace` and `dpo-jittor-cpu-10/trace`;
- `grpo-torch-cpu-10/trace` and `grpo-jittor-cpu-10/trace`;
- `logits-dpo-torch-cpu/trace`, `logits-dpo-jittor-cpu/trace`, `logits-grpo-torch-cpu/trace` and `logits-grpo-jittor-cpu/trace`;
- `dpo-torch-cuda-10/trace` and `dpo-jittor-cuda-10/trace`;
- `grpo-torch-cuda-10/trace` and corrected `grpo-jittor-cuda-10-r2/trace`;
- `logits-dpo-torch-cuda/trace`, `logits-dpo-jittor-cuda/trace`, `logits-grpo-torch-cuda/trace` and `logits-grpo-jittor-cuda/trace`;
- `dpo-cpu-summary.json`, `grpo-cpu-summary.json`, `dpo-cuda-summary.json`, `grpo-cuda-summary.json`, `final-logits-all-devices-documentation-audit.json`, `measured-source.json` and per-run launch/log receipts.

Model/environments/buffers remain in `$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/`.
External launch, summary and generation scripts are in
`$JITTOR_LAB_ROOT/trl-qwen3-sft/`; they are unversioned, not maintained repository
commands. Summaries read completed manifests without independently rehashing
every large payload; the harness checks oracle payload hashes during comparison.

| Frozen source | SHA256 |
| --- | --- |
| `trl_rl_probe.py` | `7325bb6a084a0da365073f2dbbe4dff8e1b0b24ae819a0a4a741849992c7d926` |
| `trl_dpo_case.py` | `9c212aaf32f893fb177d623f4fc062f08e9c1dc108d7997957b76fa43bc572e2` |
| `trl_grpo_case.py` | `e571eb6496e8eaa548af307ea1083cab754d6435bfd6ddd330df713a4b1e36ad` |
| `trl_sft_probe.py` | `ea63b03271bef6ef7107d451d38e58565ad85b30f3276fd1fbd5160fb6a7b245` |
| `trl_final_logits_probe.py` | `1953458c932053ea5415765be4f8c64b7f266d8c8beb753ad3f73d04bfa01f7e` |

The early GRPO smoke used different generation defaults and an older rollout
format. It is excluded from formal numerical conclusions; formal traces use
format 2 and explicit model-default control.

The exact measured scripts are retained in the unversioned `measured-sources/`
directory. After both DPO training pairs completed, only the DPO builder's
docstring was corrected to describe the actual observation scope. Its resulting
SHA256 is `55fd3f03fb4716d5290c84d917a1c4a37bf66ee315da6ef0afcd609384dacc4c`;
`dpo-docstring-only-change.json` records equality of the Python AST after
removing docstrings. The table above keeps the measured hash, not the revised
documentation hash. Reusing old oracle artifacts requires the measured script;
a fresh pair may use the corrected script on both sides.

Gradient extraction, host transfers, hashing and disk I/O dominate diagnostic
timings. The oracle retains all gradients; the candidate reads and compares
them online, keeping hashes/statistics instead of another full gradient trace.
These asymmetric workloads are **not a speed comparison**. Each oracle trace
is roughly 26 GB before filesystem overhead; budget separate methods/devices.

Final repository validation after the additional mainstream Trainer repairs
reports **98 native tests passed, 41 skipped and one expected failure**,
**224 Torch compatibility tests passed and one skipped**, **83 tooling tests
passed**, and **1,516 structure tests passed with four skips**. The final
check receipts are under `$JITTOR_LAB_ROOT/_state/trl-mainstream/`.
Interrupted or failed earlier checks are excluded. These checks validate the
resulting source tree; they do not relabel the older full-checkpoint model
measurements as having run after the later sequence-index and `tolist` repairs.
See the [mainstream report](2026-10-22-trl-mainstream.md) for those repairs and
their separate final-source model matrix.

All manifests retain `scorecard_eligible=false`. These bounded manual cases
do not grant L-level admission, nightly coverage, overall TRL compatibility,
long-run convergence, AdamW/AMP/LoRA support, distributed or checkpoint-resume
support, or performance evidence. Ten training examples cannot establish
general model quality beyond the measured objectives.
