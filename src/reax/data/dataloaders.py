from typing import Final

from . import _types, collate, fetchers, samplers

__all__ = ("GenericDataLoader",)


class GenericDataLoader(_types.DataLoader):
    def __init__(
        self,
        dataset: _types.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: _types.CollateFn | None = None,
    ):
        """Init function."""
        # Params
        self._batch_size: Final[int] = batch_size
        if collate_fn is None:
            collate_fn = collate.get_default_collator().collate
        self._collate_fn: Final[_types.CollateFn] = collate_fn

        self._dataset = dataset
        self._sampler = samplers.create_sampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    def __iter__(self):
        """Iter function."""
        fetcher = fetchers.create_fetcher(self._dataset, collate_fn=self._collate_fn)
        for indices in self._sampler:
            yield fetcher.fetch(indices)
