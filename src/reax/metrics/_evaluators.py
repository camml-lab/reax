"""Metrics evaluators"""

import abc
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
from typing_extensions import override

if TYPE_CHECKING:
    import reax

__all__ = "MetricEvaluator", "DefaultEvaluator", "VmapEvaluator"

T = TypeVar("T", bound="reax.Metric")


class MetricEvaluator(abc.ABC, eqx.Module):
    @abc.abstractmethod
    def create(self, metric: T, batch: Any, **kwargs) -> T:
        """Evaluate the metric creating a new instance"""


class DefaultEvaluator(MetricEvaluator):
    @override
    def create(self, metric: T, batch: Any, **kwargs) -> T:
        if isinstance(batch, tuple):
            return metric.create(*batch, **kwargs)

        return metric.create(batch, **kwargs)


class VmapEvaluator(MetricEvaluator):
    @override
    def create(self, metric: T, batch: Any, **kwargs) -> T:
        # 1. Compute the metric for the whole batch in parallel
        create_fn = metric.create
        if isinstance(batch, tuple):
            batched_instance = jax.vmap(create_fn)(*batch, **kwargs)
        else:
            batched_instance = jax.vmap(create_fn)(batch, **kwargs)

        # 2. Re-use the metric's own reduction logic to collapse the batch dim
        return batched_instance.reduce(axis=0)
