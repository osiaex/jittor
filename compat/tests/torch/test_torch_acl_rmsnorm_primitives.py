"""FP32 normalization gradients recorded against native PyTorch on Ascend."""

import unittest

import jittor as jt
import numpy as np
import torch
from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestACLRMSNormPrimitives(unittest.TestCase):
    def test_mean_backward_divides_by_reduction_size(self):
        upstream = np.array([-0.10075725615024567, 0.0012914960971102118,
                             -0.00031014697742648423], dtype=np.float32)
        expected = np.broadcast_to((upstream / np.float32(1536))[:, None], (3, 1536))
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.zeros((3, 1536), device="npu:0", requires_grad=True)
            x.mean(dim=-1).backward(torch.tensor(upstream, device=x.device))
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), expected)

    def test_rsqrt_backward_matches_native_pytorch_operation_order(self):
        values = np.array([0.7078934907913208, 0.666592001914978, 0.8382045030593872],
                          dtype=np.float32)
        upstream = np.array([0.0697721466422081, 0.3815600872039795, -0.024867134168744087],
                            dtype=np.float32)
        expected = np.array([-0.05857336148619652, -0.3505442142486572, 0.016202077269554138],
                            dtype=np.float32)
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.tensor(values, device="npu:0", requires_grad=True)
            torch.rsqrt(x).backward(torch.tensor(upstream, device=x.device))
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), expected)

    def test_residual_gradient_accumulates_newer_consumers_first(self):
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.tensor([1.0], device="npu:0", requires_grad=True)
            first = x * 1e20
            second = x * -1e20
            last = x * 1.0
            ((first + second) + last).sum().backward()
            self.assertEqual(x.grad.item(), 0.0)

    def test_grouped_missing_gradient_keeps_other_contributions(self):
        class CopySecond(torch.autograd.Function):
            @staticmethod
            def forward(ctx, first, second):
                return second.clone()

            @staticmethod
            def backward(ctx, upstream):
                return None, upstream

        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.tensor([2.0], device="npu:0", requires_grad=True)
            y = torch.tensor([4.0], device="npu:0", requires_grad=True)
            (x * 3.0 + CopySecond.apply(x, y)).sum().backward()
            self.assertEqual(x.grad.item(), 3.0)
            self.assertEqual(y.grad.item(), 1.0)

    def test_integer_square_contributes_one_pow_gradient(self):
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            for square in (lambda x: x.pow(2), lambda x: torch.pow(x, 2)):
                x = torch.tensor([1.0], device="npu:0", requires_grad=True)
                (square(x) + x * 16777216.0).sum().backward()
                self.assertEqual(x.grad.item(), 16777218.0)

    def test_softplus_native_backward(self):
        values = np.array([-40., -5., 0., 2., 20., 25.], dtype=np.float32)
        upstream = np.array([1., -2., 3., 0., -5., 6.], dtype=np.float32)
        expected = np.where(values > 20., values, np.logaddexp(0., values))
        expected_grad = np.where(values > 20., upstream, upstream / (1. + np.exp(-values)))
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.tensor(values, device="npu:0", requires_grad=True)
            out = torch.nn.functional.softplus(x)
            out.backward(torch.tensor(upstream, device=x.device))
            np.testing.assert_allclose(out.detach().cpu().numpy(), expected, atol=0, rtol=2e-6)
            np.testing.assert_allclose(x.grad.detach().cpu().numpy(), expected_grad, atol=0, rtol=2e-6)

    def test_log_softmax_native_backward(self):
        values = np.array([[1., -2., 3.], [7., 2., -6.]], dtype=np.float32)
        upstream = np.array([[0., 1., 0.], [1., -3., 2.]], dtype=np.float32)
        shifted = values - values.max(axis=1, keepdims=True)
        expected = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        expected_grad = upstream - np.exp(expected) * upstream.sum(axis=1, keepdims=True)
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x = torch.tensor(values, device="npu:0", requires_grad=True)
            out = torch.nn.functional.log_softmax(x, dim=-1)
            out.backward(torch.tensor(upstream, device=x.device))
            np.testing.assert_allclose(out.detach().cpu().numpy(), expected, atol=1e-7, rtol=2e-6)
            np.testing.assert_allclose(x.grad.detach().cpu().numpy(), expected_grad, atol=1e-7, rtol=2e-6)

    def test_linear_bias_backward(self):
        inputs = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        weights = np.arange(6, dtype=np.float32).reshape(2, 3)
        bias = np.array([3., -1.], dtype=np.float32)
        upstream = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            x, weight, b = [torch.tensor(value, device="npu:0", requires_grad=True)
                            for value in (inputs, weights, bias)]
            out = torch.nn.functional.linear(x, weight, b)
            out.backward(torch.tensor(upstream, device=x.device))
            np.testing.assert_array_equal(out.detach().cpu().numpy(), inputs @ weights.T + bias)
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), upstream @ weights)
            np.testing.assert_array_equal(weight.grad.detach().cpu().numpy(),
                                          upstream.reshape(-1, 2).T @ inputs.reshape(-1, 3))
            np.testing.assert_array_equal(b.grad.detach().cpu().numpy(), upstream.sum(axis=(0, 1)))

    def test_matmul_adjoint_with_repeated_kv_heads(self):
        from jittor.backends.acl.kernels.ops.matmul_op import mapped_matmul

        rng = np.random.default_rng(17)
        left_values = rng.standard_normal((2, 8, 4, 422), dtype=np.float32)
        value_heads = rng.standard_normal((2, 1, 422, 256), dtype=np.float32)
        seeds = rng.standard_normal((2, 8, 4, 256), dtype=np.float32)
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            left = torch.tensor(left_values, device="npu:0", requires_grad=True)
            value = torch.tensor(value_heads, device=left.device, requires_grad=True)
            repeated = value.expand(2, 8, 422, 256)
            upstream = torch.tensor(seeds, device=left.device)
            product = torch.matmul(left, repeated)
            actual, grad_value = torch.autograd.grad(product, (left, value), upstream)
            expected = mapped_matmul(upstream, repeated.transpose(-1, -2).contiguous())
            expected_value = mapped_matmul(left.detach(), upstream, trans_a=True).sum(1, keepdim=True)
            np.testing.assert_array_equal(actual.detach().cpu().numpy(),
                                          expected.detach().cpu().numpy())
            np.testing.assert_allclose(grad_value.detach().cpu().numpy(),
                                       expected_value.detach().cpu().numpy(), atol=1e-5, rtol=1e-4)
