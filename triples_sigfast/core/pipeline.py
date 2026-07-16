"""
triples_sigfast.core.pipeline
-----------------------------
Lazy generator architecture for chunked signal processing.

Provides the SigPipeline class, which allows chaining of operations
(map, filter) on out-of-core datasets yielding data in chunks.
"""

from collections.abc import Callable, Iterable

__all__ = ["SigPipeline"]


class SigPipeline:
    """
    Lazy pipeline for chunked data processing.

    Wraps an iterable (like a chunked file reader) and allows chaining
    operations lazily. Operations are only evaluated when the pipeline
    is iterated or compute() is called.
    """

    def __init__(self, source: Iterable):
        self._source = source

    def __iter__(self):
        return iter(self._source)

    def map(self, func: Callable, *args, **kwargs) -> "SigPipeline":
        """Apply a function to each chunk lazily."""

        def _generator():
            for chunk in self._source:
                yield func(chunk, *args, **kwargs)

        return SigPipeline(_generator())

    def filter(self, predicate: Callable) -> "SigPipeline":
        """
        Apply a boolean mask filter to each chunk lazily.
        The predicate should return a boolean array/mask for the chunk.
        """

        def _generator():
            for chunk in self._source:
                mask = predicate(chunk)
                yield chunk[mask]

        return SigPipeline(_generator())

    def compute(self) -> list:
        """
        Evaluate the pipeline and return the result as a list of chunks.
        """
        return list(self)
