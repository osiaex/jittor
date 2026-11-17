"""Gradient mode and gradient clipping.

Automatic mixed precision used to live here too; it is now owned by
:mod:`jittor.compat.torch.amp` and :mod:`jittor.compat.torch.grad_scaler`, and
re-exported at the bottom of this module so every historical import path keeps
resolving to the same objects.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from typing import Any, Dict, List, cast
import jittor as jt
from .. import collectives as _collectives


import functools as _functools


class _GradDecoratorCtx:
    """Mimics torch.no_grad/enable_grad: usable as a context manager, a bare
    decorator (@torch.no_grad), and a called decorator (@torch.no_grad())."""

    def __init__(self, scope_factory, func=None):
        self._scope_factory = scope_factory
        self._func = func if callable(func) else None

    def __call__(self, *args, **kwargs):
        # used as @torch.no_grad() returning a decorator, then applied to a func
        if self._func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            @_functools.wraps(func)
            def wrapped(*a, **k):
                with self._scope_factory():
                    return func(*a, **k)
            return wrapped
        # used as @torch.no_grad (bare): self._func was set at construction
        if self._func is not None:
            with self._scope_factory():
                return self._func(*args, **kwargs)
        raise TypeError("no_grad/enable_grad misuse")

    def __get__(self, obj, objtype=None):
        # Descriptor protocol: when @torch.no_grad wraps a *method*, this instance
        # replaces the method in the class dict. Without __get__, `inst.method`
        # returns this object unbound, so `self` is never passed and the first
        # real arg wrongly binds to the method's `self` (e.g. transformers'
        # @torch.no_grad ConversionOps.convert -> "missing 'input_dict'"). Bind
        # the instance like a normal function descriptor. Plain context-manager
        # instances (no wrapped func) are never class attributes -> return self.
        if self._func is None or obj is None:
            return self
        import types as _types
        return _types.MethodType(self, obj)

    def __enter__(self):
        self._scope = self._scope_factory()
        return self._scope.__enter__()

    def __exit__(self, *exc):
        return self._scope.__exit__(*exc)




def _reduce_norm_group(value, how, group):
    if group is None or group.size() == 1:
        return value
    if group.ranks is None:
        return _collectives._reduce_scalar(value, how)
    if group.rank() < 0:
        raise RuntimeError("gradient norm called by a process-group nonmember")
    kind = group._get_backend_name()
    ops = getattr(jt.compile_extern, kind + "_ops", None)
    gather = getattr(ops, kind + "_all_gather", None)
    if gather is None or group._backend_handle is None:
        raise RuntimeError("gradient norm requires the mesh communicator")
    gathered = gather(value.reshape((1,)), group._backend_handle)
    return getattr(gathered, how)()


def _get_total_norm_device(grads, norm_type=2.0, error_if_nonfinite=False,
                           shard_reduce=False):
    """Compute the total norm for a list of gradient Vars on device.

    ``shard_reduce`` says these gradients are *shards*: each rank holds a
    different slice of the same logical gradient, so the norm has to be
    combined across ranks before the root is taken. Without it every rank
    clipped by its own slice's norm -- always smaller than the true one -- so
    each rank scaled by a different, too-large coefficient and the training
    trajectory silently diverged from torch's.

    It must stay off for DDP, where every rank already holds the *same*
    averaged gradient: reducing there would count the same norm N times.
    """
    import math as _math

    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return jt.array(0.0)

    # A singleton group has no distributed reduction and must retain the
    # ordinary per-tensor dtype and rounding semantics.
    if shard_reduce is not True and shard_reduce and shard_reduce.size() == 1:
        shard_reduce = False

    # Each shard set has its own communicator. Replicated gradients and
    # ordinary parameters are counted once; combining all through WORLD
    # would duplicate the replica axis and mix independent meshes.
    if shard_reduce is True and any(hasattr(g, "_fsdp_norm_group") for g in grads):
        groups: Dict[Any, List[Any]] = {}
        for grad in grads:
            group = getattr(grad, "_fsdp_norm_group", None)
            groups.setdefault(group, []).append(grad)
        norms = [_get_total_norm_device(values, norm_type, False,
                                       shard_reduce=group if group is not None else False)
                 for group, values in groups.items()]
        p = float(norm_type)
        if len(norms) == 1:
            total = norms[0]
        else:
            values = jt.concat([norm.reshape((1,)) for norm in norms])
            if p == float("inf"):
                total = values.max()
            elif p == float("-inf"):
                total = values.min()
            elif p == 0:
                total = values.sum()
            else:
                total = (values ** p).sum() ** (1 / p)
        if error_if_nonfinite and not _math.isfinite(float(total.item())):
            raise RuntimeError("The total norm for gradients is non-finite")
        return total

    def _across(value, how):
        if not shard_reduce:
            return value
        if shard_reduce is not True:
            return _reduce_norm_group(value, how, shard_reduce)
        return _collectives._reduce_scalar(value, how)
    p = float(norm_type)
    dtypes = [_jittor_dtype_name(g.dtype) for g in grads]
    real_dtypes = [{"complex64": "float32", "complex128": "float64"}.get(d, d)
                   for d in dtypes]
    if "float64" in real_dtypes:
        result_dtype = "float64"
    elif "float32" in real_dtypes or len(set(real_dtypes)) > 1:
        result_dtype = "float32"
    else:
        result_dtype = real_dtypes[0]
    acc_dtype = "float64" if result_dtype == "float64" else "float32"

    # A locally empty tensor has no identity for these norms. Shards can
    # legitimately be empty, so leave their identities for the collective.
    if not shard_reduce and (p < 0 or p == float("inf")):
        if any(int(g.numel()) == 0 for g in grads):
            raise RuntimeError("linalg.vector_norm cannot compute this norm on an empty tensor")

    if p == 0.0:
        # torch first computes each tensor's zero-norm, then the zero-norm of
        # those scalars: this counts tensors containing at least one nonzero.
        nonempty = []
        for g in grads:
            x = g.abs() if "complex" in _jittor_dtype_name(g.dtype) else g
            nonempty.append((_across((x != 0).sum(), "max") != 0).reshape((1,)))
        total = jt.concat(nonempty).sum().cast(acc_dtype)
        if not shard_reduce:
            total = total.cast(result_dtype)
    else:
        # Reduce each gradient before combining, avoiding a model-sized
        # temporary. Torch rounds each tensor's norm before combining them.
        parts = []
        for g, real_dtype in zip(grads, real_dtypes):
            x = g.abs() if "complex" in _jittor_dtype_name(g.dtype) else g
            local_dtype = acc_dtype if shard_reduce else (
                "float64" if real_dtype == "float64" else "float32")
            x = x.cast(local_dtype)
            ax = x.abs()
            # Torch's vector norm of a singleton is its absolute value, even
            # when a power would overflow or underflow.
            singleton = not shard_reduce and int(x.numel()) == 1
            if singleton:
                value = ax.sum()
            elif p == float("inf"):
                value = ax.max() if int(x.numel()) else jt.array(float("-inf")).cast(acc_dtype)
            elif p == float("-inf"):
                value = ax.min() if int(x.numel()) else jt.array(float("inf")).cast(acc_dtype)
            elif p == 1.0:
                value = ax.sum()
            elif p == 2.0:
                value = (x * x).sum()
            else:
                value = (ax ** p).sum()
            if not shard_reduce:
                if not singleton and p == 2.0:
                    value = jt.sqrt(value)
                elif not singleton and p not in (1.0, float("inf"), float("-inf")):
                    value = value ** (1.0 / p)
                value = value.cast(real_dtype).cast(acc_dtype)
            parts.append(value.reshape((1,)))
        if not shard_reduce and len(parts) == 1:
            # The outer vector norm also has one element; avoid a second
            # power and root which could overflow or underflow.
            total = parts[0]
        else:
            values = jt.concat(parts)
            if p == float("inf"):
                total = _across(values.max(), "max")
            elif p == float("-inf"):
                total = _across(values.min(), "min")
            elif p == 1.0:
                total = _across(values.sum(), "sum")
            elif p == 2.0:
                total = jt.sqrt(_across(
                    values.sum() if shard_reduce else (values * values).sum(), "sum"))
            else:
                total = _across(
                    values.sum() if shard_reduce else (values ** p).sum(), "sum") ** (1.0 / p)
        if not shard_reduce:
            total = total.cast(result_dtype)

    if error_if_nonfinite:
        total_value = float(total.item())
        if not _math.isfinite(total_value):
            raise RuntimeError(
                "The total norm of order %s for gradients is non-finite, so it "
                "cannot be clipped. To disable this error set "
                "error_if_nonfinite=False." % norm_type
            )
    return total


def _clip_grads_with_norm_device(grads, max_norm, total_norm):
    """Scale gradient Vars using an already-computed total norm."""
    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return

    dtype = _jittor_dtype_name(total_norm.dtype)
    limit = float(max_norm)
    # Torch computes the scalar coefficient in the norm's dtype, including
    # the addition and division rounding for float16 and bfloat16.
    # Native jt.array otherwise defaults to auto_convert_64_to_32, which
    # rounds these constants before a later cast back to float64.
    epsilon = jt.array(1e-6, dtype="float64").cast(dtype)
    max_norm_value = jt.array(limit, dtype="float64").cast(dtype)
    denom = (total_norm + epsilon).cast(dtype)
    raw_coef = (max_norm_value / denom).cast(dtype)
    coef = jt.minimum(cast(Any, raw_coef), jt.array(1.0).cast(dtype))
    # CUDA fmin-style minimum may select the finite operand for NaN. Torch
    # propagates a NaN total norm into every gradient when errors are disabled.
    coef = jt.ternary(jt.isnan(raw_coef), raw_coef, coef)
    for g in grads:
        grad_dtype = _jittor_dtype_name(g.dtype)
        if "complex" in grad_dtype:
            g.update(g * coef.cast(grad_dtype))
        else:
            # Neither operand may be narrowed before multiplication: the
            # public clip_grads_with_norm_ accepts an arbitrary norm dtype.
            if grad_dtype == "float64" or dtype == "float64":
                compute_dtype = "float64"
            elif grad_dtype == "float32" or dtype == "float32" or grad_dtype != dtype:
                compute_dtype = "float32"
            else:
                compute_dtype = dtype
            g.update((g.cast(compute_dtype) * coef.cast(compute_dtype)).cast(grad_dtype))


def _clip_grad_norm_device(grads, max_norm, norm_type=2.0,
                           error_if_nonfinite=False, shard_reduce=False):
    """Clip a list of gradient Vars without a host-side coefficient branch.

    Reductions operate on individual gradients to avoid a model-sized
    temporary. The device coefficient avoids a per-step D2H synchronization.
    """
    grads = [g for g in grads if isinstance(g, jt.Var)]
    total = _get_total_norm_device(grads, norm_type, error_if_nonfinite,
                                   shard_reduce=shard_reduce)
    _clip_grads_with_norm_device(grads, max_norm, total)
    return total


#: The automatic-mixed-precision family moved to :mod:`jittor.compat.torch.amp`
#: when its stubs were replaced by real implementations (this module has an
#: 800-line budget and the family is ~800 lines on its own). Every historical
#: import path keeps working: these are the same objects, not copies.
from .amp import (                     # noqa: E402, F401
    OptState,
    autocast,
    autocast_cache_enabled,
    autocast_configured_dtype,
    autocast_decorator,
    autocast_decrement_nesting,
    autocast_dtype,
    autocast_increment_nesting,
    autocast_is_enabled,
    autocast_nesting,
    amp_definitely_not_available,
    clear_autocast_cache,
    cuda_custom_bwd,
    cuda_custom_fwd,
    custom_bwd,
    custom_fwd,
    is_autocast_available,
    set_autocast_cache_enabled_state,
    set_autocast_dtype_state,
    set_autocast_enabled_state,
    GradScaler,
    _amp_cast,
    _AutocastContext,
    _CpuAutocast,
    _CpuGradScaler,
    _CudaAutocast,
    _CudaGradScaler,
    _GradScaler,
)
