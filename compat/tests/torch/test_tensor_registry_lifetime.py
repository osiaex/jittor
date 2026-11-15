"""Autograd registration follows live Tensor holders and native leaf identity."""

import pytest

from _helpers.child_process import run_python_child


@pytest.mark.parametrize("case", (
    "nonleaf", "disconnected", "retained", "collection", "interleaved",
    "unowned_parameter", "retain_contract", "stale_view", "stale_view_with_optimizer",
))
def test_independent_autograd_registration_lifetime(case):
    script = """
import gc
import sys
import weakref
import numpy as np
from jittor.compat.shim import activate
torch = activate()["torch"]
import jittor as jt
use_accelerator = jt.introspection.policy.runtime.use_cuda
device = ('npu' if torch.npu.is_available() else 'cuda') if use_accelerator else 'cpu'
x = torch.tensor([2.], device=device, requires_grad=True)
if use_accelerator:
    x.sync()
    assert x.location() == "device", "CUDA probe silently used host storage"
case = sys.argv[1]
if case == "nonleaf":
    middle = x * 3
    middle.requires_grad_(True)
    middle.sum().backward()
    assert middle.grad is None, "requires_grad_ registered a non-leaf"
    np.testing.assert_array_equal(x.grad.numpy(), [3.])
elif case == "disconnected":
    other = torch.tensor([4.], requires_grad=True)
    other.square().sum().backward()
    assert x.grad is None
    x.square().sum().backward()
    assert x.grad is not None, "unrelated backward erased a live leaf"
    np.testing.assert_array_equal(x.grad.numpy(), [4.])
elif case == "retained":
    middle = x * 3
    middle.retain_grad()
    loss = middle.sum()
    loss.backward(retain_graph=True)
    loss.backward(retain_graph=True)
    np.testing.assert_array_equal(middle.grad.numpy(), [2.])
elif case == "collection":
    middle = x * 3
    middle.retain_grad()
    references = (weakref.ref(x), weakref.ref(middle))
    del x, middle
    gc.collect()
    assert all(ref() is None for ref in references), "registry retained dead holders"
elif case == "interleaved":
    first, second = x * 2, x * 3
    first.retain_grad()
    second.retain_grad()
    first.sum().backward(retain_graph=True)
    second.sum().backward()
    np.testing.assert_array_equal(second.grad.numpy(), [1.])
elif case == "unowned_parameter":
    owned = torch.nn.Parameter(torch.tensor([4.]))
    unowned = torch.nn.Parameter(torch.tensor([5.]))
    optimizer = torch.optim.SGD([owned], lr=0.1)
    owned.square().sum().backward()
    unowned.square().sum().backward()
    np.testing.assert_array_equal(unowned.grad.numpy(), [10.])
elif case in ("stale_view", "stale_view_with_optimizer"):
    from jittor.compat.torch.tensor_state import get_tensor_state
    with torch.no_grad():
        stale = torch.Tensor(x)
        x.copy_(x)
    assert stale.requires_grad
    assert not stale._has_leaf_accumulator()
    # Simulate a live registry entry whose alias history changed after registration.
    get_tensor_state(jt).leaf_params[id(stale)] = stale
    if case == "stale_view_with_optimizer":
        x = torch.nn.Parameter(torch.tensor([2.], device=device))
        optimizer = torch.optim.SGD([x], lr=0.1)
    x.square().sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [4.])
    try:
        stale.clone()
    except RuntimeError as error:
        assert "no_grad mode" in str(error)
    else:
        raise AssertionError("using a stale view must still raise")
elif case == "retain_contract":
    assert x.retain_grad() is None and not x.retains_grad
    middle = x * 3
    assert middle.retain_grad() is None and middle.retains_grad
    try:
        torch.tensor([1.]).retain_grad()
    except RuntimeError as error:
        assert "requires_grad=False" in str(error)
    else:
        raise AssertionError("retain_grad accepted a non-differentiable Tensor")
print("REGISTRY_LIFETIME_OK")
"""
    result = run_python_child(
        ["-c", script, case], without_torch_mode=True, merge_stderr=True,
    )
    assert result.returncode == 0, result.stdout
    assert "REGISTRY_LIFETIME_OK" in result.stdout
