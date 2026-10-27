"""Unit tests for :mod:`toy_bald.data.batching`."""

from __future__ import annotations

import pytest
import torch

from toy_bald.data.batching import (
    LengthBucketBatchSampler,
    bucket_boundaries,
    bucket_for,
)

BOUNDARIES = (4, 8, 16)
LENGTHS = [1, 3, 5, 7, 9, 15, 2, 4, 6, 16, 8, 12, 3]


def bucket_of_batch(batch: list[int]) -> set[int]:
    """
    :param batch:   Row indices of one batch.
    :return:        The distinct buckets of the batch's rows.
    """
    return {bucket_for(LENGTHS[index], BOUNDARIES) for index in batch}


def test_bucket_boundaries_ends_at_max_length_and_drops_larger_candidates() -> None:
    """
    Given: The default candidates 32..512 and a token budget of 200.
    When:  The boundaries are computed.
    Then:  Candidates below 200 are kept and 200 itself closes the list.
    """
    assert bucket_boundaries(200) == (32, 64, 128, 200)
    assert bucket_boundaries(512) == (32, 64, 128, 256, 512)


def test_bucket_boundaries_raises_value_error_for_non_positive_budget() -> None:
    """
    Given: A token budget of zero.
    When:  The boundaries are computed.
    Then:  A ValueError is raised instead of returning an unusable bucket list.
    """
    with pytest.raises(ValueError, match="positive"):
        bucket_boundaries(0)


def test_bucket_for_returns_smallest_boundary_that_fits_inclusive() -> None:
    """
    Given: Boundaries (4, 8, 16).
    When:  Lengths at and just past each boundary are bucketed.
    Then:  A length equal to a boundary stays in it, and one past moves up.
    """
    assert [bucket_for(length, BOUNDARIES) for length in (1, 4, 5, 8, 9, 16)] == [
        4,
        4,
        8,
        8,
        16,
        16,
    ]


def test_bucket_for_raises_value_error_when_length_exceeds_largest_bucket() -> None:
    """
    Given: Boundaries ending at 16.
    When:  A 17-token prompt is bucketed.
    Then:  A ValueError is raised, since the prompt should have been filtered out.
    """
    with pytest.raises(ValueError, match="exceeds"):
        bucket_for(17, BOUNDARIES)


def test_sampler_without_shuffle_yields_every_index_once_in_bucket_and_index_order() -> None:
    """
    Given: A sampler over mixed lengths with no shuffling and no dropping.
    When:  One epoch is drawn.
    Then:  Every index appears exactly once, each batch stays within one bucket, and
           indices keep their order within a bucket.
    """
    sampler = LengthBucketBatchSampler(
        LENGTHS, BOUNDARIES, batch_size=2, shuffle=False, drop_last=False
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(LENGTHS)))
    assert all(len(bucket_of_batch(batch)) == 1 for batch in batches)
    assert batches[0] == [0, 1]
    assert len(batches) == len(sampler)


def test_sampler_with_drop_last_yields_only_full_batches() -> None:
    """
    Given: Buckets of 5, 4 and 4 rows and a batch size of 3, with drop_last.
    When:  One shuffled epoch is drawn.
    Then:  Each bucket contributes only full batches, so every batch has 3 rows and
           the reported length matches.
    """
    generator = torch.Generator().manual_seed(0)
    sampler = LengthBucketBatchSampler(
        LENGTHS, BOUNDARIES, batch_size=3, shuffle=True, drop_last=True, generator=generator
    )

    batches = list(sampler)

    assert len(batches) == len(sampler) == 3
    assert all(len(batch) == 3 for batch in batches)
    assert all(len(bucket_of_batch(batch)) == 1 for batch in batches)


def test_sampler_replays_the_same_epoch_after_restoring_generator_state() -> None:
    """
    Given: A shuffling sampler whose generator state is saved before an epoch.
    When:  The epoch is drawn, the state restored, and the epoch drawn again.
    Then:  Both draws match, which is what resuming from a checkpoint relies on, and
           the next epoch differs from the first.
    """
    generator = torch.Generator().manual_seed(7)
    sampler = LengthBucketBatchSampler(
        LENGTHS, BOUNDARIES, batch_size=2, shuffle=True, drop_last=False, generator=generator
    )
    state = generator.get_state()

    first = list(sampler)
    generator.set_state(state)
    replay = list(sampler)
    following = list(sampler)

    assert first == replay
    assert following != first


def test_sampler_raises_value_error_for_non_positive_batch_size() -> None:
    """
    Given: A batch size of zero.
    When:  The sampler is constructed.
    Then:  A ValueError is raised.
    """
    with pytest.raises(ValueError, match="batch_size"):
        LengthBucketBatchSampler(LENGTHS, BOUNDARIES, batch_size=0, shuffle=False, drop_last=False)
