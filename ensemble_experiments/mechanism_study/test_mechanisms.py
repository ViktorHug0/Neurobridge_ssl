"""Checks for interventions, fusion ceilings, and shared/frozen gradient paths."""
import unittest
import numpy as np
import torch
from .common import margin_metrics
from .score_analysis import rearrange, convex_oracle
from .train_pair import ARMS, make_pair


class MechanismsTest(unittest.TestCase):
    def test_interventions_preserve_true_ranks(self):
        rng = np.random.default_rng(12)
        scores = rng.normal(size=(3, 12, 12))
        truth = np.arange(12)
        ranks = (scores > scores[:, truth, truth, None]).sum(-1)
        for mode in ['aligned', 'permuted', 'stratified']:
            changed = rearrange(scores, mode, rng, rng.normal(size=(12, 12)))
            np.testing.assert_array_equal(changed[:, truth, truth], scores[:, truth, truth])
            np.testing.assert_array_equal((changed > changed[:, truth, truth, None]).sum(-1), ranks)
        self.assertAlmostEqual(margin_metrics(rearrange(scores, 'aligned', rng))['distractor_bonus'], 0.)

    def test_convex_oracle_can_exceed_member_oracle(self):
        scores = np.array([[[.8, .9, .1], [.1, .9, .2], [.1, .2, .9]],
                           [[.8, .1, .9], [.1, .9, .2], [.1, .2, .9]]])
        self.assertAlmostEqual(margin_metrics(scores)['oracle_top1'], 200 / 3)
        self.assertEqual(convex_oracle(scores), 100.)
        scores[:, 0, 0] = -1
        self.assertAlmostEqual(convex_oracle(scores), 200 / 3)

    def test_temporal_mask_removes_outside_information(self):
        model, _ = make_pair(ARMS['temporal'], [32, 32], 63, 3300)
        model.eval()
        x, images = torch.randn(3, 63, 250), torch.randn(3, 64)
        subject = torch.ones(3, dtype=torch.long)
        with torch.no_grad():
            a = model(x, images, subject)
            changed = x.clone(); changed[:, :, 150:] += 100
            b = model(changed, images, subject)
        for first, second in zip(a, b):
            torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)

    def test_shared_stem_is_evaluated_once_and_both_heads_receive_gradients(self):
        model, _ = make_pair(ARMS['shared_stem'], [32, 32], 63, 3300)
        before = model.stem[2].num_batches_tracked.clone()
        result = model(torch.randn(4, 63, 250), torch.randn(4, 64), torch.ones(4, dtype=torch.long))
        sum(e.square().mean() for e, _ in result).backward()
        self.assertEqual(int(model.stem[2].num_batches_tracked - before), 1)
        self.assertGreater(float(model.stem[0].weight.grad.abs().sum()), 0)
        for head in model.eeg_heads:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))

    def test_frozen_branch_has_no_gradients_or_bn_updates(self):
        model, _ = make_pair(ARMS['frozen_b030'], [32, 32], 63, 3300)
        model.train(); model.freeze_first()
        before = {k: v.clone() for k, v in model.encoders[0].state_dict().items()}
        result = model(torch.randn(4, 63, 250), torch.randn(4, 64), torch.ones(4, dtype=torch.long))
        sum(e.square().mean() for e, _ in result).backward()
        for name, tensor in model.encoders[0].state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.encoders[0].parameters()))

    def test_shared_full_parameter_matching(self):
        independent, _ = make_pair(ARMS['independent_targets'], [4096, 4096], 63, 3300)
        shared, _ = make_pair(ARMS['shared_full_matched'], [4096, 4096], 63, 3300)
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertLess(abs(n(shared) / n(independent) - 1), .005)


if __name__ == '__main__':
    unittest.main()
