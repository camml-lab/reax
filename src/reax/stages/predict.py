from typing import TYPE_CHECKING, Any, Optional, Union

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import stages

if TYPE_CHECKING:
    import reax


__all__ = ("Predict",)


class Predict(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = float("inf"),
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "Predicting", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )
        self._all_outputs = []

    @override
    def _step(self) -> "reax.stages.MetricResults":
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any):
        """On iteration finishing."""
        self._all_outputs.append(outputs)
