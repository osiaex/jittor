"""Gradient clipping and AMP-scaler correctness/synchronization regressions."""

from _helpers import capability as _test_capability

import math
import unittest

import numpy as np
import jittor as jt
import torch
from jittor.compat.torch import _GradScaler, _clip_grad_norm_device
from jittor.compat.torch.grad import _clip_grads_with_norm_device


_DEVICES = [("cpu", 0)] + ([ ("cuda", 1) ] if _test_capability.any_accelerator_enabled(backend=jt) else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class _FakeOptimizer:
    def __init__(self, grads):
        self.param_groups = [{"grads": grads}]
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return self.steps


class TestGradientManagement(unittest.TestCase):
    def test_autograd_grad_unused_semantics(self):
        def make_graph():
            # torch.autograd.grad, so torch tensors: ``grad`` and the
            # requires-grad bookkeeping it reads live on the torch frontend.
            used = torch.tensor(np.array([1.0, 2.0], dtype="float32"),
                                requires_grad=True)
            unused = torch.tensor(np.array([3.0, 4.0], dtype="float32"),
                                  requires_grad=True)
            return used, unused, (used * used).sum()

        used, unused, loss = make_graph()
        grads = torch.autograd.grad(loss, (used, unused), allow_unused=True)
        self.assertIsNotNone(grads[0])
        self.assertIsNone(grads[1])

        used, unused, loss = make_graph()
        with self.assertRaisesRegex(RuntimeError, "allow_unused=True"):
            torch.autograd.grad(loss, (used, unused))

        used, unused, loss = make_graph()
        grads = torch.autograd.grad(
            loss, (used, unused), materialize_grads=True)
        np.testing.assert_array_equal(grads[1].numpy(), np.zeros(2, dtype="float32"))

        used, unused, loss = make_graph()
        with self.assertRaisesRegex(ValueError, "allow_unused"):
            torch.autograd.grad(
                loss, (used, unused), allow_unused=False,
                materialize_grads=True)

        used, unused, loss = make_graph()
        grads = torch.autograd.grad(
            loss, (used, unused), create_graph=True,
            materialize_grads=True)
        self.assertTrue(grads[1].requires_grad)

    def test_clip_grad_norm_values_and_no_item(self):
        def body(dev):
            for p, expected in ((1.0, 10.0), (2.0, math.sqrt(34.0)),
                                (float("inf"), 4.0), (3.0, 118.0 ** (1.0 / 3.0))):
                grads = [jt.array(np.array([3.0, 4.0], dtype="float32")),
                         jt.array(np.array([0.0, -3.0], dtype="float32"))]
                calls = []
                original = jt.Var.item

                def counted(var, *args, **kwargs):
                    calls.append(1)
                    return original(var, *args, **kwargs)

                jt.Var.item = counted
                try:
                    total = _clip_grad_norm_device(grads, 1.0, p)
                    values = jt.fetch_sync([total] + grads)
                finally:
                    jt.Var.item = original
                self.assertEqual(len(calls), 0, f"clip_grad_norm_ synced on {dev}, p={p}")
                self.assertAlmostEqual(float(values[0].reshape(-1)[0]), expected, places=5)
                clipped = np.concatenate([values[1].reshape(-1), values[2].reshape(-1)])
                if p == float("inf"):
                    got_norm = np.abs(clipped).max()
                else:
                    got_norm = np.linalg.norm(clipped, ord=p)
                self.assertLessEqual(float(got_norm), 1.00001)

        both_devices(body)

    def test_clip_grad_norm_does_not_concat_gradient_payloads(self):
        # Catch a model-sized temporary using only small CPU gradients.
        with jt.flag_scope(use_cuda=0):
            for p in (0.0, 1.0, 2.0, 3.5, -1.0, float("inf"), float("-inf")):
                grads = [jt.array(np.full(64, 2.0, dtype="float32")),
                         jt.array(np.full(32, 4.0, dtype="float32"))]
                original_concat = jt.concat

                def checked_concat(values, *args, **kwargs):
                    values = list(values)
                    self.assertLessEqual(sum(int(v.numel()) for v in values), 8)
                    return original_concat(values, *args, **kwargs)

                jt.concat = checked_concat
                try:
                    total = _clip_grad_norm_device(grads, 1.0, p)
                finally:
                    jt.concat = original_concat
                if p == 0:
                    expected = 2.0
                elif p == float("inf"):
                    expected = 4.0
                elif p == float("-inf"):
                    expected = 2.0
                else:
                    expected = (64 * 2.0 ** p + 32 * 4.0 ** p) ** (1.0 / p)
                self.assertAlmostEqual(float(total.item()), expected, delta=abs(expected)*2e-5)

    def test_clip_grad_norm_empty_tensor_semantics(self):
        with jt.flag_scope(use_cuda=0):
            for p in (-1.0, float("inf"), float("-inf")):
                grad = jt.array(np.array([], dtype="float32"))
                with self.assertRaisesRegex(RuntimeError, "empty tensor"):
                    _clip_grad_norm_device([grad], 1.0, p)
            for p in (0.0, 1.0, 2.0):
                grad = jt.array(np.array([], dtype="float32"))
                total = _clip_grad_norm_device([grad], 1.0, p)
                self.assertEqual(float(total.item()), 0.0)

    def test_clip_grad_norm_half_and_mixed_precision(self):
        with jt.flag_scope(use_cuda=0):
            grad = jt.array(np.array([3.0, 4.0], dtype="float16"))
            total = _clip_grad_norm_device([grad], 1.0)
            self.assertEqual(str(total.dtype), "float16")
            self.assertEqual(float(total.item()), 5.0)
            np.testing.assert_array_equal(
                grad.numpy(), np.array([0.599609375, 0.7998046875], dtype="float16"))

            grads = [jt.array(np.array([0.101, 0.202], dtype="float16")),
                     jt.array(np.array([0.303], dtype="float32"))]
            total = _clip_grad_norm_device(grads, 0.1)
            self.assertEqual(str(total.dtype), "float32")
            self.assertAlmostEqual(float(total.item()), 0.37789976596832275, places=6)

            grads = [jt.array(np.array([0.49, 0.89, 0.93, 1.0], dtype="float16")),
                     jt.array(np.array([1.0], dtype="float32"))]
            _clip_grad_norm_device(grads, 0.24372333230597973)
            np.testing.assert_array_equal(
                grads[0].numpy(), np.array(
                    [0.06048583984375, 0.10986328125, 0.11480712890625,
                     0.12347412109375], dtype="float16"))

    def test_clip_grad_norm_bfloat16_and_complex128(self):
        with jt.flag_scope(use_cuda=0):
            grad = jt.array(np.array([3.0, 4.0], dtype="float32")).cast("bfloat16")
            total = _clip_grad_norm_device([grad], 1.0)
            self.assertEqual(str(total.dtype), "bfloat16")
            self.assertEqual(float(total.item()), 5.0)
            np.testing.assert_array_equal(
                grad.float32().numpy(), np.array([0.6015625, 0.80078125]))

            grad = jt.array(np.array([3.0 + 4.0j], dtype="complex128"),
                            dtype=torch.complex128)
            total = _clip_grad_norm_device([grad], 1.0)
            self.assertEqual(str(total.dtype), "float64")
            self.assertEqual(float(total.item()), 5.0)
            np.testing.assert_allclose(
                grad.numpy(), np.array([3.0 + 4.0j]) / 5.000001, rtol=1e-12)

    def test_clip_grad_norm_nonfinite_limit(self):
        with jt.flag_scope(use_cuda=0):
            for limit, value in ((float("inf"), float("inf")),
                                 (float("inf"), float("nan")),
                                 (float("nan"), 1.0)):
                grad = jt.array(np.array([1.0, value], dtype="float32"))
                _clip_grad_norm_device([grad], limit)
                self.assertTrue(np.isnan(grad.numpy()).all())

    def test_clip_grad_norm_single_element_extremes(self):
        with jt.flag_scope(use_cuda=0):
            for p in (-3.0, -1.0, 0.5, 2.0, 3.5,
                      float("inf"), float("-inf")):
                for value in (1e20, 1e-23, 0.0, np.inf, np.nan):
                    grad = jt.array(np.array([value], dtype="float32"))
                    total = _clip_grad_norm_device([grad], 1.0, p)
                    actual = float(total.item())
                    expected = float(np.float32(abs(value)))
                    if np.isnan(expected):
                        self.assertTrue(np.isnan(actual), (p, value))
                    else:
                        self.assertEqual(actual, expected, (p, value))

    def test_clip_grad_norm_singleton_group_dtype(self):
        class SingletonGroup:
            def size(self):
                return 1

        with jt.flag_scope(use_cuda=0):
            grad = jt.array(np.array([3.0, 4.0], dtype="float16"))
            object.__setattr__(grad, "_fsdp_norm_group", SingletonGroup())
            total = _clip_grad_norm_device([grad], 1.0, shard_reduce=True)
            self.assertEqual(str(total.dtype), "float16")
            np.testing.assert_array_equal(
                grad.numpy(), np.array([0.599609375, 0.7998046875], dtype="float16"))

    def test_clip_grads_with_supplied_lower_precision_norm(self):
        with jt.flag_scope(use_cuda=0):
            grad = jt.array(np.array([1e100], dtype="float64"), dtype="float64")
            total = jt.array(np.array(2.0, dtype="float16"), dtype="float16")
            self.assertEqual(str(grad.dtype), "float64")
            np.testing.assert_array_equal(grad.numpy(), np.array([1e100], dtype="float64"))
            _clip_grads_with_norm_device([grad], 1.0, total)
            self.assertEqual(str(grad.dtype), "float64")
            np.testing.assert_allclose(grad.numpy(), [5e99], rtol=1e-15)

    def test_clip_grad_norm_float64_scalar_precision(self):
        with jt.flag_scope(use_cuda=0):
            grad = jt.array(np.array([3.0, 4.0], dtype="float64"), dtype="float64")
            total = _clip_grad_norm_device([grad], 1.0)
            self.assertEqual(str(total.dtype), "float64")
            np.testing.assert_allclose(
                grad.numpy(), np.array([3.0, 4.0]) / 5.000001, rtol=1e-15)

    def test_clip_grad_norm_nonfinite_error(self):
        def body(dev):
            grad = jt.array(np.array([1.0, np.inf], dtype="float32"))
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                _clip_grad_norm_device([grad], 1.0, 2.0, error_if_nonfinite=True)

        both_devices(body)

    def test_clip_grad_norm_nonfinite_propagates_when_allowed(self):
        def body(dev):
            for value in (np.inf, np.nan):
                grad = jt.array(np.array([1.0, value], dtype="float32"))
                total = _clip_grad_norm_device(
                    [grad], 1.0, 2.0, error_if_nonfinite=False)
                total_np, grad_np = jt.fetch_sync([total, grad])
                self.assertFalse(np.isfinite(total_np).all(), f"total {value} on {dev}")
                if np.isnan(value):
                    self.assertTrue(np.isnan(grad_np).all(), f"NaN propagation on {dev}")
                else:
                    self.assertEqual(float(grad_np[0]), 0.0, f"Inf finite entry on {dev}")
                    self.assertTrue(np.isnan(grad_np[1]), f"Inf*0 on {dev}")

        both_devices(body)

    def test_clip_grad_norm_zero_and_negative_inf(self):
        def body(dev):
            for p, expected in ((0.0, 2.0), (float("-inf"), 0.0)):
                grads = [jt.array(np.array([3.0, 0.0], dtype="float32")),
                         jt.array(np.array([4.0], dtype="float32"))]
                total = _clip_grad_norm_device(grads, float("inf"), p)
                got = float(total.item())
                self.assertEqual(got, expected, f"p={p} on {dev}")

        both_devices(body)

    def test_grad_scaler_uses_one_item_and_skips_inf_step(self):
        def body(dev):
            finite = [jt.array(np.array([8.0, -4.0], dtype="float32")),
                      jt.array(np.array([2.0], dtype="float32"))]
            opt = _FakeOptimizer(finite)
            scaler = _GradScaler(init_scale=2.0)
            scaler.scale(torch.tensor(1.0, device=dev))
            calls = []
            original = jt.Var.item

            def counted(var, *args, **kwargs):
                calls.append(1)
                return original(var, *args, **kwargs)

            jt.Var.item = counted
            try:
                scaler.unscale_(opt)
            finally:
                jt.Var.item = original
            self.assertEqual(len(calls), 1, f"GradScaler item count on {dev}")
            self.assertFalse(scaler._found_inf)
            self.assertEqual(scaler.step(opt), 1)
            np.testing.assert_allclose(finite[0].numpy(), [4.0, -2.0], atol=1e-6)

            bad = _FakeOptimizer([
                jt.array(np.array([2.0, np.inf], dtype="float32")),
                jt.array(np.array([np.nan], dtype="float32")),
            ])
            scaler = _GradScaler(init_scale=2.0)
            scaler.scale(torch.tensor(1.0, device=dev))
            scaler.unscale_(bad)
            self.assertTrue(scaler._found_inf)
            self.assertIsNone(scaler.step(bad))
            self.assertEqual(bad.steps, 0)

        both_devices(body)


if __name__ == "__main__":
    unittest.main()
