# TRL SFTTrainer: full Qwen3-0.6B checkpoint

- Status: CPU and CUDA complete three training updates; strict numerical acceptance fails on both (378 and 3 fields respectively). This is a bounded manual experiment, not maintained nightly admission.
- Reviewed: 2026-10-21.
- Owner: Torch compatibility and downstream verification maintainers.
- Source: `2.0-refactor` baseline `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`, plus the integral-sum compatibility fix in this change.
- Review when: model revision, downstream versions, loss/reduction implementations, precision, probe or acceptance budgets change.

## Scope

The official `trl.SFTTrainer` executes its original training loop and the official
Transformers Qwen3 model. There is no replacement Trainer, miniature model,
frozen layer, LoRA adapter or handwritten optimizer loop. The complete pretrained
checkpoint has 28 layers and **596,049,920 unique trainable parameters** in 310
tensors; embedding and LM head weights are tied.

| Item | Frozen choice |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| TRL | 0.21.0; official source `46d09bd2408f17605409fb3ee8ba12705add7faa` |
| Transformers / Accelerate / datasets | 4.56.2 / 1.10.1 / 3.6.0 |
| Independent oracle | Genuine PyTorch 2.8.0+cu128 in a separate environment |
| Candidate | Jittor 2.0 core and its current Torch compatibility shim; no genuine Torch runtime |
| Precision / attention | FP32 / eager; no AMP or gradient checkpointing |
| Training | Three SGD updates, learning rate 0.0001, no momentum, decay or clipping |
| Input | One fixed text repeated, batch 1, maximum length 32, actual input shape `(1,15)` |
| Data loading | Map-style dataset, zero workers, no pinning or packing |
| Platform | Python 3.11.17, g++ 11.4.0; CUDA 12.8.93, RTX 2080 Ti (11 GB, SM75) |

The short repeated sample checks the training machinery, not fine-tuning quality
or convergence. Its raw text and actual token IDs, masks, labels and Trainer loss
denominator are retained. Integer input IDs do not require gradients; the probe
instead observes the embedding-output gradient and every trainable parameter's
gradient. Every update is followed by a complete parameter snapshot.

Both runtimes load identical parameter bytes. A shared oracle-created artifact
also supplies nonpersistent buffers, including RoPE frequencies, whose separate
initialization otherwise differs slightly. The probe checks all initial hashes,
installed TRL package bytes against the clean official source, input metadata,
complete observation names, artifact hashes and finite values. Jittor executes
with backend fallback forbidden; every observed tensor's physical residency and
native dispatch agree with the requested CPU or CUDA device.

## Findings

The first complete comparison exposed a real compatibility defect:
`labels.ne(-100).sum()` returned int32 instead of Torch's int64. The Trainer's
`num_items_in_batch` therefore had the wrong dtype. The fix promotes integral
inputs **before** summation, preventing intermediate int32 overflow, and honors
explicit `dtype` and `out`. It belongs to the shared Tensor compatibility owner,
not a TRL-specific adapter. CPU and CUDA each pass 16 focused cases, with the
contracts also checked against genuine Torch on both devices.

The original streaming-dataset attempt stops at missing `Tensor.share_memory_`.
The measured map-style dataset avoids that requirement; it does not establish
streaming or shared-worker support. This limitation is tracked as KI-COMPAT-016
in the [issue ledger](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/known-issues.md).

| Device / runtime | Step 1 loss | Step 2 loss | Step 3 loss | Updates / fallback |
| --- | ---: | ---: | ---: | --- |
| CPU / Torch | 3.108144999 | 2.385542154 | 1.995426297 | 3 / 0 |
| CPU / Jittor | 3.108174562 | 2.385568142 | 1.995449305 | 3 / 0 |
| CUDA / Torch | 3.108174086 | 2.385566473 | 1.995449066 | 3 / 0 |
| CUDA / Jittor | 3.108175039 | 2.385566473 | 1.995448828 | 3 / 0 |

The frozen comparator requires each floating element to satisfy
`abs(candidate-reference) <= 1e-5 + 1e-4 * max(abs(reference), field_max*1e-6, 1e-30)`.
Inputs are exact. Each device has 1,881 tensor pairs. These budgets have not been
relaxed after observing failures.

**CPU fails this numerical criterion:** 378 fields fail (372 parameter gradients,
three logits and three embedding-output gradients). The largest absolute
gradient discrepancy is `5.692e-4`. All 930 updated parameter fields pass; their
largest absolute discrepancy is `2.384e-7`. Matching updates with this small
learning rate does not override the gradient failures.

**CUDA also fails the unchanged criterion, but only for its three logits
fields:** 352 elements exceed the budget, with maximum absolute logit error
`3.719e-5`. All 930 parameter-gradient fields, all 930 updated parameter fields,
all three embedding-output gradients and all three losses pass. Their maximum
absolute errors are respectively `4.196e-5`, `5.960e-8`, `7.182e-6` and
`9.537e-7`; the combined absolute/relative rule explains why a larger gradient
absolute error can pass while a smaller logit error fails. The exact input and
initial-state checks also pass. These remaining logit differences have not been
fully attributed, so the report does not claim complete numerical parity.

Regression checks: the probe's 17 offline tests pass; integral-sum tests pass
16 cases on CPU and 16 on CUDA. The maintained core file selections run in
separate CPU processes: native has 98 passed, 41 skipped and one expected
failure; Torch mode has 224 passed and one skipped. The complete structure
selection initially has 1,514 passed, four skipped and two failures: missing
`asv==0.6.6` in the test environment and the new report missing from generated
`MANIFEST.in`. After installing the declared test dependency and regenerating
the manifest, both affected files pass all 18 tests. The full structure
selection is not repeated. Layout, generated-manifest, whitespace and the
changed Python owners' Ruff checks pass. Raw logs retain the original failures
and the targeted rerun; none is relabeled as an original clean run.

Read-only diagnostics identify one numerical source, without claiming complete
root-cause attribution. Recomputing cross entropy from each runtime's saved
logits in NumPy FP64 gives a first-step cross-runtime loss difference of
`7.87e-7`, versus the observed FP32 difference of `2.956e-5`. Genuine Torch's
FP32 cross entropy on the same saved logits reproduces its recorded loss and
differs from the FP64 analytic logits gradient by relative L2 `2.464e-5`.
The independent oracle therefore also contributes measurable rounding error.
This does not establish that all failing model gradients are explained by it.

## Reproduction and evidence

Use the [full-checkpoint SFT probe instructions](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md)
with separate genuine-Torch and Jittor environments. Run the oracle first to
create `--initial-buffers`, reuse the same model and buffer artifact, and compare
matching devices. A nonzero comparison exit is the expected CPU result for
the strict budget above. Each runtime's three-step trace requires about 14 GB.
For the recorded runs, add `--threads 4` to both `run` commands and set
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`.
The candidate uses `JT_CUDA_KERNEL_MATH=strict`,
`JT_USE_PARALLEL_OP_COMPILER=0` and `DISABLE_MULTIPROCESSING=1`, with a separate
`JITTOR_HOME` for each device. Preserve the original default comparison budgets;
CUDA also exits nonzero for the three logit fields above.

Unversioned evidence lives under `$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/`:

- `torch-cpu-fixed-3steps/trace` and `jittor-cpu-sumfix-3steps/trace`;
- `torch-cuda-fixed-3steps/trace` and `jittor-cuda-sumfix-retry-3steps/trace`;
- `cpu-sumfix-3steps-comparison.json` and `cuda-sumfix-3steps-comparison.json`;
- `model-info.json`, `model-revision.txt`, `downstream-lock.txt`, `initial-buffers.npz`;
- `measured-source.json`, `measured-sum-fix.patch`, focused regression logs and launch receipts.
- `cpu_numeric_diagnostic.py` and `cpu_numeric_diagnostic.json`, the reproducible read-only rounding analysis.

The measured probe SHA256 is
`ea63b03271bef6ef7107d451d38e58565ad85b30f3276fd1fbd5160fb6a7b245`;
the sum-owner SHA256 is
`2d404bd2c1dc25d54b7cef4cfa8ab006e69736fe1f2be4e129dfac9006606324`.
Receipts bind the dirty measurement source to these files; this is not an
exhaustive imported-file or wheel-archive provenance claim.

An earlier CUDA attempt was manually interrupted after a long silent first
step; it is retained as incomplete. The subsequent complete run reused the
compiled cache and added an external periodic traceback observer. It completed
all updates, including the ordinary Trainer logging path. This does not prove
the interrupted run was deadlocked.

Snapshot I/O, JIT compilation and other activity on the shared host preclude a
throughput comparison. This experiment does not cover AdamW, mixed precision,
long sequences, checkpoint saving/resume, streaming workers, distributed SFT,
other TRL algorithms or newer dependency versions. `scorecard_eligible` remains
false; running three steps is not a claim of full TRL compatibility.
