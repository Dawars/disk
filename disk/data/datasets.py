import itertools, os
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import DataLoader, Sampler

from disk.data import DISKDataset


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None, reinit=None, generator=None) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.reinit = reinit
        self.generator = generator
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        self.reinit(self.data_source)
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

def get_datasets(
        root,
        no_depth=None,
        batch_size=2,
        crop_size=(768, 768),
        substep=1,
        n_epochs=50,
        chunk_size=5000,
        train_limit=1000,
        test_limit=250,
):
    if no_depth is None:
        raise ValueError("Unspecified no_depth")

    train_dataset = DISKDataset(
        os.path.join(root, 'train/dataset.json'),
        crop_size=crop_size,
        limit=train_limit,
        shuffle=True,
        no_depth=no_depth,
    )
    dataloader_kwargs = {
        'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': min(batch_size, 12),
    }

    train_chunk_iter = RandomSampler(
        train_dataset,
        num_samples=chunk_size,
        reinit=lambda dataset: dataset.shuffle(),
        generator=None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        # shuffle=True,
        batch_size=batch_size,
        sampler=train_chunk_iter,
        **dataloader_kwargs
    )

    test_dataset = DISKDataset(
        os.path.join(root, 'test/dataset.json'),
        crop_size=crop_size,
        limit=test_limit,
        shuffle=True,
        no_depth=no_depth,
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False,
        batch_size=batch_size, **dataloader_kwargs
    )

    return train_dataloader, test_dataloader


class DividedIter:
    def __init__(self, iterable, n_repeats=1, n_chunks=None,
                 chunk_size=None, reinit=None):

        if (n_chunks is None) == (chunk_size is None):
            raise ValueError(
                'Exactly one of `n_chunks` and `chunk_size` has to be None'
            )

        self._iterable = iterable
        self._base_length = len(iterable)

        if chunk_size is None:
            chunk_size = self._base_length // n_chunks
        if n_chunks is None:
            n_chunks = self._base_length // chunk_size

        self.n_repeats = n_repeats
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.reinit = reinit

        self.total_chunks = self.n_chunks * self.n_repeats

    def __len__(self):
        return self.total_chunks

    def __iter__(self):
        for _ in range(self.n_repeats):
            if self.reinit is not None:
                self.reinit(self._iterable)

            base_iter = iter(self._iterable)

            for _ in range(self.n_chunks):
                yield itertools.islice(base_iter, self.chunk_size)
