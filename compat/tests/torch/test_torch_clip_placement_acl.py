"""Host-side scalar clipping inside an Ascend module."""
import unittest

import jittor as jt
import torch
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator('acl', backend=jt).enabled, 'No ACL found')
class TestClipPlacementACL(unittest.TestCase):
    def test_host_scalar_bounds_keep_input_placement(self):
        class VisionIndices(torch.nn.Module):
            def forward(self, device_input):
                positions = torch.linspace(0, 7, 5, device='cpu').int() + 1
                return positions.clip(max=7), device_input.clip(min=0, max=7)

        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback='error'), torch.no_grad():
            before = jt.core.backend_fallback_count()
            x = torch.tensor([-2, 4, 10], device='npu:0', dtype=torch.int32)
            host, device = VisionIndices()(x)
            self.assertEqual(host.device.type, 'cpu')
            self.assertEqual(host.tolist(), [1, 2, 4, 6, 7])
            self.assertEqual(device.device.type, 'npu')
            self.assertEqual(device.tolist(), [0, 4, 7])
            self.assertEqual(jt.core.backend_fallback_count(), before)
