import unittest
import numpy as np
import torch
from .models import ARMS, CompactDecoder
from .train import split_indices, capture_rng, restore_rng
from train import build_image_positive_mask


class Designs(unittest.TestCase):
    def test_parameters_and_alignment(self):
        baseline = CompactDecoder('single').size()['parameters']
        for arm in ARMS:
            model = CompactDecoder(arm)
            self.assertEqual(model.align_dims, [128] * len(model.targets))
            self.assertLessEqual(model.size()['parameters'], baseline * 1.05)

    def test_shapes_gradients_and_shared_execution(self):
        for arm in ARMS:
            model = CompactDecoder(arm)
            calls = []
            handle = model.backbone.tsconv[0].register_forward_hook(lambda *args: calls.append(1))
            eeg = torch.randn(9, 63, 250)
            images = torch.randn(9, 6400)
            features = model(eeg, images)
            self.assertEqual(len(calls), 1)
            handle.remove()
            for e, i in features:
                self.assertEqual(tuple(e.shape), (9, 128))
                self.assertEqual(tuple(i.shape), (9, 128))
            positives = build_image_positive_mask(torch.arange(9)//3, torch.zeros(9, dtype=torch.long))
            loss = model.loss(features, positives)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_single_matches_original_path(self):
        model = CompactDecoder('single').eval()
        x = torch.randn(3, 63, 250)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.encode(x)[0], model.eeg_heads[0](model.backbone(x))))

    def test_concept_split(self):
        class Data:
            def get_image_group_indices(self):
                return {(c, i): [c*90+i*9+s for s in range(9)] for c in range(1654) for i in range(10)}
        tr, va, held = split_indices(Data())
        expected = sorted(np.random.default_rng(20260822).permutation(range(1654))[:165].tolist())
        self.assertEqual(held, expected)
        self.assertEqual(len(tr)+len(va), 1654*90)
        self.assertFalse(set(tr) & set(va))
        self.assertFalse({i//90 for i in tr} & set(held))
        self.assertEqual({i//90 for i in va}, set(held))

    def test_state_roundtrip(self):
        for arm in ARMS:
            model = CompactDecoder(arm).eval()
            other = CompactDecoder(arm).eval()
            other.load_state_dict(model.state_dict())
            x = torch.randn(2, 63, 250)
            with torch.no_grad():
                for a, b in zip(model.encode(x), other.encode(x)):
                    self.assertTrue(torch.equal(a, b))


if __name__ == '__main__': unittest.main()
