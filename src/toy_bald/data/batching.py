"""
Length-bucketed batching for statically compiled models.

``torch.compile(..., dynamic=False)`` builds one graph per input shape. Padding every
batch to the global maximum keeps a single shape but wastes most of the compute on
padding, because the median prompt is far shorter than the maximum. Padding each batch
to the smallest of a few fixed lengths ("buckets") keeps the number of graphs small and
the padding modest. Grouping prompts of similar length into the same batch, as
:class:`LengthBucketBatchSampler` does for training, shrinks the padding further.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Sequence
from typing import cast

import torch
from torch.utils.data import Sampler

DEFAULT_BUCKETS: tuple[int, ...] = (32, 64, 128, 256, 512)


def bucket_boundaries(
    max_length: int, candidates: Sequence[int] = DEFAULT_BUCKETS
) -> tuple[int, ...]:
    """
    Return the padded lengths used for a token budget of ``max_length``.

    :param max_length:  Longest prompt allowed, in tokens; always the last boundary.
    :param candidates:  Increasing bucket lengths; those not below ``max_length`` are dropped.
    :return:            Strictly increasing boundaries ending at ``max_length``.
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    return (*(length for length in sorted(set(candidates)) if length < max_length), max_length)


def bucket_for(length: int, boundaries: Sequence[int]) -> int:
    """
    Return the smallest boundary that fits ``length`` tokens.

    :param length:      Token count of the longest prompt in a batch.
    :param boundaries:  Output of :func:`bucket_boundaries`.
    :return:            The padded length.
    :raises ValueError: If ``length`` exceeds the last boundary.
    """
    position = bisect.bisect_left(boundaries, length)
    if position == len(boundaries):
        raise ValueError(f"length {length} exceeds the largest bucket {boundaries[-1]}")
    return boundaries[position]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    Yield batches of indices whose prompts all fall in the same length bucket.

    Each epoch the indices of every bucket are shuffled and cut into batches, and the
    order of the batches across buckets is shuffled, so consecutive batches still mix
    lengths. With ``drop_last`` every batch has exactly ``batch_size`` rows, which keeps
    one compiled graph per bucket; the dropped remainder changes every epoch.

    The shuffles draw from ``generator`` when :meth:`__iter__` starts, so saving and
    restoring the generator state between epochs resumes the same batch order.

    :param lengths:     Token count per dataset row.
    :param boundaries:  Output of :func:`bucket_boundaries`.
    :param batch_size:  Rows per batch.
    :param shuffle:     Shuffle within buckets and across batches; otherwise keep index order.
    :param drop_last:   Drop each bucket's final incomplete batch.
    :param generator:   Source of randomness for the shuffles.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        boundaries: Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # Group the row indices by bucket, in index order within each bucket.
        self.buckets: dict[int, list[int]] = {boundary: [] for boundary in boundaries}
        for index, length in enumerate(lengths):
            self.buckets[bucket_for(length, boundaries)].append(index)

    def _num_batches(self, size: int) -> int:
        """
        :param size:    Rows in one bucket.
        :return:        Batches that bucket contributes.
        """
        if self.drop_last:
            return size // self.batch_size
        return -(-size // self.batch_size)

    def _permutation(self, size: int) -> list[int]:
        """
        :param size:    Number of items to permute.
        :return:        A random permutation of ``range(size)`` drawn from the generator.
        """
        permutation = torch.randperm(size, generator=self.generator)
        return cast(list[int], permutation.tolist())  # pyright: ignore[reportUnknownMemberType]

    def __len__(self) -> int:
        return sum(self._num_batches(len(indices)) for indices in self.buckets.values())

    def __iter__(self) -> Iterator[list[int]]:
        # Cut every bucket into batches, shuffling the bucket's rows first when requested.
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            if self.shuffle:
                indices = [indices[position] for position in self._permutation(len(indices))]
            for batch_index in range(self._num_batches(len(indices))):
                start = batch_index * self.batch_size
                batches.append(indices[start : start + self.batch_size])

        # Interleave the buckets so a run of batches does not share one length.
        if self.shuffle:
            batches = [batches[position] for position in self._permutation(len(batches))]
        yield from batches
