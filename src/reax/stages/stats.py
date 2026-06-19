"""Stage for evaluating dataset statistics"""

from typing import TYPE_CHECKING

import beartype
from flax import nnx
import jaxtyping as jt
from typing_extensions import override

from reax import metrics

from . import stages

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
        engine: "reax.Engine",
        *,
        rngs: nnx.Rngs,
        dataset_name: str = "train",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        ignore_missing=False,
        evaluator: "reax.metrics.MetricEvaluator | None" = None,
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
        self._evaluator = evaluator
        self._stats = stats
        self._ignore_missing = ignore_missing

    @override
    def _step(self) -> None:
        """Step function."""
        evaluator = (
            self._evaluator if self._evaluator is not None else self._engine.metric_evaluator
        )
        collection = metrics.MetricCollection(self._stats, evaluator)

        # Calculate...
        if isinstance(self.batch, tuple):
            calculated = collection.create(*self.batch, ignore_missing=self._ignore_missing)
        else:
            calculated = collection.create(self.batch, ignore_missing=self._ignore_missing)

        # ...and log all the stats
        for name, stat in calculated.items():
            self.log(name, stat, on_step=False, on_epoch=True, logger=True)
