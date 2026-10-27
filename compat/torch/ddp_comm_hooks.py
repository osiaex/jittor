"""Import boundaries for unsupported custom DDP communication hooks.

Accelerate resolves these names even when its hook selection is NO. Default
DDP reduction remains owned by the existing reducer; custom hook execution
must refuse rather than silently dropping a requested communication policy.
"""
from ..transaction import current_transaction
from .fidelity import Fidelity, register_api_bindings


def reject_registration(*args, **kwargs):
    raise NotImplementedError(
        "torch.nn.parallel.DistributedDataParallel.register_comm_hook: custom "
        "DDP communication hooks are not implemented; use the default DDP reduction")


def _unsupported(name):
    def reject(*args, **kwargs):
        raise NotImplementedError(
            name + ": custom DDP communication hooks are not implemented; "
            "use the default DDP reduction")
    reject.__name__ = name.rsplit('.', 1)[-1]
    reject.__qualname__ = reject.__name__
    return reject


_PREFIX = 'torch.distributed.algorithms.ddp_comm_hooks'
fp16_compress_hook = _unsupported(_PREFIX + '.default_hooks.fp16_compress_hook')
bf16_compress_hook = _unsupported(_PREFIX + '.default_hooks.bf16_compress_hook')
fp16_compress_wrapper = _unsupported(_PREFIX + '.default_hooks.fp16_compress_wrapper')
bf16_compress_wrapper = _unsupported(_PREFIX + '.default_hooks.bf16_compress_wrapper')
powerSGD_hook = _unsupported(_PREFIX + '.powerSGD_hook.powerSGD_hook')
batched_powerSGD_hook = _unsupported(_PREFIX + '.powerSGD_hook.batched_powerSGD_hook')


class PowerSGDState:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            _PREFIX + '.powerSGD_hook.PowerSGDState: PowerSGD communication is not implemented')


def install_ddp_comm_hooks(registry):
    """Publish canonical modules through the active installation registry."""
    modules = registry.module_map
    transaction = current_transaction()
    before = dict(modules) if transaction is not None else None
    published = dict(registry._published) if transaction is not None else None
    parents = [modules[name] for name in ('torch.distributed.algorithms', _PREFIX,
               _PREFIX + '.default_hooks', _PREFIX + '.powerSGD_hook') if name in modules]
    attributes = [(parent, dict(vars(parent))) for parent in parents]
    try:
        registry.ensure(_PREFIX, package=True)
        for suffix, bindings in (
            ('default_hooks', dict(fp16_compress_hook=fp16_compress_hook,
                bf16_compress_hook=bf16_compress_hook, fp16_compress_wrapper=fp16_compress_wrapper,
                bf16_compress_wrapper=bf16_compress_wrapper)),
            ('powerSGD_hook', dict(PowerSGDState=PowerSGDState, powerSGD_hook=powerSGD_hook,
                batched_powerSGD_hook=batched_powerSGD_hook)),
        ):
            name = _PREFIX + '.' + suffix
            module = registry.ensure(name)
            for key, value in bindings.items():
                setattr(module, key, value)
            register_api_bindings(module, name, bindings, Fidelity.UNIMPLEMENTED,
                'Importable for default NO-hook DDP; custom communication hooks explicitly refuse execution')
    finally:
        if transaction is not None:
            for parent, attrs in attributes:
                transaction.record_object_diffs(parent, attrs)
            transaction.record_mapping_diffs(registry._modules, before)
            transaction.record_mapping_diffs(registry._published, published)
