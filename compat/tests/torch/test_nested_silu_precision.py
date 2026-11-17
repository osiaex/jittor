"""A nested module must retain SiLU's opmath after its first publication."""
import unittest

import numpy as np
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class _NestedSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.SiLU()

    def forward(self, value):
        return self.activation(value)


class TestNestedSiLU(cu.JittorTestCase):
    def test_fp32_cuda_autocast_preserves_silu_forward_and_gradient(self, device):
        if str(device).split(":")[0] != "cuda":
            self.skipTest("CUDA autocast precision")
        source = np.array([-6., -3.125, -1.3125, -.171875, .625, 1.6875, 4.], dtype=np.float32)
        sigmoid = 1. / (1. + np.exp(-source.astype(np.float64)))
        expected = source * sigmoid
        derivative = sigmoid * (1. + source * (1. - sigmoid))
        with forbid_backend_fallbacks():
            for amp_dtype in (torch.float16, torch.bfloat16):
                for operation in (torch.nn.functional.silu, _NestedSiLU().to(device)):
                    x = torch.tensor(source, dtype=torch.float32, device=device, requires_grad=True)
                    with torch.autocast(device, dtype=amp_dtype):
                        y = operation(x)
                    self.assertEqual(str(y.dtype), str(torch.float32))
                    gradient, = torch.autograd.grad(y.sum(), (x,))
                    np.testing.assert_allclose(y.detach().cpu().numpy(), expected, atol=1e-7, rtol=2e-6)
                    np.testing.assert_allclose(gradient.detach().cpu().numpy(), derivative, atol=1e-7, rtol=2e-6)

    def test_first_and_warm_calls_keep_functional_opmath(self, device):
        rng = np.random.RandomState(17831)
        shape = (2, 4, 17)
        source = (np.linspace(-7, 7, np.prod(shape)).reshape(shape)
                  + rng.randn(*shape) * .13).astype(np.float32)
        cotangent = (rng.randn(*shape) * .7).astype(np.float32)
        with forbid_backend_fallbacks():
            for precision in (torch.float16, torch.bfloat16, torch.float32):
                for enabled in (False, True):
                    module = _NestedSiLU().to(device)
                    amp_dtype = torch.float16 if precision == torch.float32 else precision
                    reference_x = torch.nn.Parameter(torch.tensor(source, device=device).to(precision))
                    with torch.autocast(device, dtype=amp_dtype, enabled=enabled):
                        reference = torch.nn.functional.silu(reference_x)
                    seed = torch.tensor(cotangent, device=device)
                    (reference.float() * seed).sum().backward()
                    expected = reference.detach().float().cpu().numpy()
                    expected_grad = reference_x.grad.detach().float().cpu().numpy()
                    for call in range(3):
                        with self.subTest(precision=str(precision), enabled=enabled, call=call):
                            x = torch.nn.Parameter(torch.tensor(source, device=device).to(precision))
                            with torch.autocast(device, dtype=amp_dtype, enabled=enabled):
                                y = module(x)
                            (y.float() * seed).sum().backward()
                            for value in (y, x.grad):
                                value.sync()
                                self.assertEqual(str(value.dtype), str(precision))
                                self.assertEqual(value.location(), 'cpu' if device == 'cpu' else 'device')
                            np.testing.assert_array_equal(y.detach().float().cpu().numpy(), expected)
                            np.testing.assert_array_equal(x.grad.detach().float().cpu().numpy(), expected_grad)


instantiate_device_type_tests(TestNestedSiLU, globals())

if __name__ == '__main__':
    unittest.main()
