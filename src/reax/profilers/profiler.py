import abc
from collections.abc import Generator
import contextlib
import logging

from typing_extensions import override

_LOGGER = logging.getLogger(__name__)

__all__ = "Profiler", "DummyProfiler"


class Profiler(abc.ABC):
    """The base class for REAX profilers."""

    @abc.abstractmethod
    def start(self, profile_name: str, **kwargs):
        """Start a new profile"""

    @abc.abstractmethod
    def stop(self, profile_name: str):
        """Stops an existing profile"""

    @contextlib.contextmanager
    def profile(self, profile_name: str, **kwargs):
        self.start(profile_name, **kwargs)
        try:
            yield
        finally:
            self.stop(profile_name)

    @abc.abstractmethod
    def start_recording(self, action_name: str) -> None:
        """Start recording an action."""

    @abc.abstractmethod
    def stop_recording(self, action_name: str) -> None:
        """Stop recording a previously started action."""

    @contextlib.contextmanager
    def profile_action(self, action_name: str) -> Generator:
        """Yields a context manager to encapsulate the scope of a profiled action.

        Example::

            with self.profile('load training data'):
                # load training data code

        The profiler will start once you've entered the context and will automatically
        stop once you exit the code block.

        """
        self.start_recording(action_name)
        try:
            yield action_name
        finally:
            self.stop_recording(action_name)


class DummyProfiler(Profiler):
    """The no-op profiler that allows you to avoid conditions like:


        if profiler is not None:
            ...

    and keep your code clean.
    """

    @override
    def start(self, profile_name: str, **kwargs):
        """Start a new profile"""

    @override
    def stop(self, profile_name: str):
        """Stops an existing profile"""

    @override
    def start_recording(self, action_name: str) -> None:
        """Start recording an action."""

    @override
    def stop_recording(self, action_name: str) -> None:
        """Stop recording a previously started action."""
