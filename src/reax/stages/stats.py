"""Stage for evaluating dataset statistics"""

from typing import TYPE_CHECKING

import beartype
from flax import nnx
import jaxtyping as jt
from typing_extensions import override

from reax import metrics

from . import stages
from .. import exceptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    import reax


__all__ = ("EvaluateStats",)


class EvaluateStats(stages.EpochStage):
    """A stage that can be used to evaluate statistics (in the form of metrics) on a dataset"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        stats: "reax.Metric | Sequence[reax.Metric] | dict[str, reax.Metric]",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine | None",
        *,
        rngs: nnx.Rngs,
        dataset_name: str = "train",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        ignore_missing=False,
        evaluator: "reax.metrics.MetricEvaluator" = None,
    ):
        """Init function."""
        super().__init__(
            "stats",
            None,
            datamanager,
            engine,
            rngs=rngs,
            dataloader_name=dataset_name,
            limit_batches=limit_batches,
            fast_dev_run=fast_dev_run,
        )

        # Params
        self._stats = metrics.MetricCollection(stats)
        self._ignore_missing = ignore_missing
        self._evaluator = evaluator if evaluator is not None else metrics.DefaultEvaluator()

    @override
    def _step(self) -> None:
        """Step function."""
        # Calculate and log all the stats
        for name, stat in self._stats.items():
            try:
                calculated = self._evaluator.create(stat, self.batch)
                self.log(name, calculated, on_step=False, on_epoch=True, logger=True)

            except exceptions.DataNotFound:
                if not self._ignore_missing:
                    raise
