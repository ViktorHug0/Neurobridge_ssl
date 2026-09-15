import unittest

import numpy as np

from ensemble_experiments.balanced_subject_bagging import NUM_BAGS, bags_for_target
from ensemble_experiments.retrieval_fusion import (
    FUSION_METHODS,
    fuse_scores,
    oracle_top1,
    retrieval_accuracies,
)


class RetrievalEnsembleTest(unittest.TestCase):
    def test_balanced_subject_bags_are_deterministic_and_nearly_uniform(self):
        for target in range(1, 11):
            bags = bags_for_target(target)
            self.assertEqual(bags, bags_for_target(target))
            self.assertEqual(len(bags), NUM_BAGS)
            self.assertEqual(len({tuple(bag) for bag in bags}), NUM_BAGS)
            self.assertTrue(all(len(bag) == 7 and target not in bag for bag in bags))
            counts = {
                subject: sum(subject in bag for bag in bags)
                for subject in range(1, 11)
                if subject != target
            }
            self.assertEqual(max(counts.values()) - min(counts.values()), 1)

    def test_fusion_rules_return_query_candidate_scores(self):
        scores = np.asarray(
            [
                [[3.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]],
                [[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]],
            ],
            dtype=np.float32,
        )
        for method in FUSION_METHODS:
            fused = fuse_scores(scores, method)
            self.assertEqual(fused.shape, (3, 3))
            self.assertTrue(np.isfinite(fused).all())
            self.assertEqual(retrieval_accuracies(fused, topk=3), (100.0, 100.0))
        self.assertEqual(oracle_top1(scores), 100.0)

    def test_row_z_prevents_one_members_scale_from_dominating(self):
        scores = np.asarray(
            [
                [[0.9, 0.1, 0.0]],
                [[0.0, 100.0, 99.0]],
                [[0.8, 0.1, 0.0]],
            ],
            dtype=np.float32,
        )
        self.assertEqual(fuse_scores(scores, "raw").argmax(axis=1).item(), 1)
        self.assertEqual(fuse_scores(scores, "row_z").argmax(axis=1).item(), 0)

    def test_fusion_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "unknown fusion method"):
            fuse_scores(np.zeros((2, 3, 3), dtype=np.float32), "learned_on_test")


if __name__ == "__main__":
    unittest.main()
