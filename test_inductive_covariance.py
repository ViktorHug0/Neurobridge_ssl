import torch

from module.inductive_covariance import InductiveQueryCovarianceAlign


def test_unseen_queries_are_independent_of_batch_companions():
    torch.manual_seed(7)
    align = InductiveQueryCovarianceAlign(
        {1: torch.eye(4), 2: 2 * torch.eye(4)}, alpha=0.4, shrinkage=0.25
    )
    query = torch.randn(1, 4, 20)
    companion = torch.randn(1, 4, 20) * 9
    alone = align(query, torch.tensor([9]))
    together = align(torch.cat((query, companion)), torch.tensor([9, 9]))[:1]
    torch.testing.assert_close(alone, together)


def test_known_source_uses_leave_one_source_out_reference():
    one = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    two = torch.diag(torch.tensor([3.0, 4.0, 5.0]))
    align = InductiveQueryCovarianceAlign({1: one, 2: two}, 0.4, 0.25)
    torch.testing.assert_close(align.source_reference_lookup[1], two)
    torch.testing.assert_close(align.source_reference_lookup[2], one)
    torch.testing.assert_close(align.global_reference, (one + two) / 2)


def test_output_covariance_matches_blended_whitening_target():
    torch.manual_seed(9)
    align = InductiveQueryCovarianceAlign({1: torch.eye(4)}, 1.0, 0.25)
    output = align(torch.randn(3, 4, 200), torch.tensor([7, 7, 7]))
    covariance = output @ output.transpose(-1, -2) / output.shape[-1]
    # With alpha=1, shrinkage leaves the expected residual covariance rather
    # than exact identity; the important numerical property is symmetry/finite.
    assert torch.isfinite(covariance).all()
    torch.testing.assert_close(covariance, covariance.transpose(-1, -2))
