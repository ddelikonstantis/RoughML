import concurrent
import logging
import os
import pickle
import statistics
import time
from abc import ABC, abstractmethod
from functools import wraps

import torch
from pyinsect.collector.NGramGraphCollector import (
    ArrayGraph2DCollector,
    HPG2DCollector,
    NGramGraphCollector,
)

from roughml.content.quantization import KBinsDiscretizerQuantizer
from roughml.shared.configuration import Configuration

logger = logging.getLogger(__name__)


def per_row(method=None, *, expected_ndim=2):
    """The following decorator, given a multidimensional matrix, applies
    the decorated function on every row of the provided matrix and returns
    a one dimensional matrix, consisting of the accumulated return values
    of all the calls **or** a singular value, in case the multidimensional
    matrix has less than expected dimensions.
    """

    def wrapper(method):
        @wraps(method)
        def wrapper_wrapper(self, matrix, *args, **kwargs):
            if len(matrix.shape) > expected_ndim:
                return torch.tensor(
                    [method(self, row, *args, **kwargs) for row in matrix]
                )

            return method(self, matrix, *args, **kwargs)

        return wrapper_wrapper

    return wrapper if method is None else wrapper(method)


class ContentLoss(Configuration, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = KBinsDiscretizerQuantizer(**kwargs)

        self.surfaces = self.quantizer.surfaces

    def __len__(self):
        return len(self.surfaces)

    @abstractmethod
    def __call__(self, surface):
        return self.quantizer(surface)

    def __str__(self):
        return str({"shape": self.surfaces.shape})

    def to_pickle(self, path):
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path):
        with path.open("rb") as file:
            return pickle.load(file)


class NGramGraphContentLoss(ContentLoss):
    """An `n-gram graph` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.surfaces = self.surfaces.reshape(self.surfaces.shape[0], -1)

        self._collector = NGramGraphCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    @per_row(expected_ndim=1)
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface).reshape(-1))


class ArrayGraph2DContentLoss(ContentLoss):
    """A `2D array graph` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._collector = ArrayGraph2DCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))


class HPG2DContentLoss(ContentLoss):
    """A `Hierarchical Proximity Graph (HPG)` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._build_collector()

    def _build_collector(self):
        total_start_time = time.time()

        self._collector = HPG2DCollector()

        logger.info("Constructing %02d graphs", len(self.surfaces))

        elapsed_time = []
        for index, surface in enumerate(self.surfaces):
            start_time = time.time()
            self._collector.add(surface)
            elapsed_time.append(time.time() - start_time)

            logger.info("Constructed graph %02d in %07.3fs", index, elapsed_time[-1])

        logger.info(
            "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
            len(self.surfaces),
            time.time() - total_start_time,
            statistics.mean(elapsed_time),
            statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
        )

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))


class HPG2DParallelContentLoss(HPG2DContentLoss):
    def _build_collector(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
            total_start_time = time.time()

            self._collector = HPG2DCollector()

            futures = {}
            for index, surface in enumerate(self.surfaces):
                future = pool.submit(self._collector.add, surface)

                futures[future] = (index, time.time())

            logger.info("Awaiting %02d jobs", len(self.surfaces))

            elapsed_time = [None] * len(self.surfaces)
            for future in concurrent.futures.as_completed(futures):
                self._collector._add_graph(future.result())

                index, start_time = futures[future]

                elapsed_time[index] = time.time() - start_time

                logger.info(
                    "Job %02d completed after %07.3fs", index, elapsed_time[index]
                )

            logger.info(
                "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
                len(self.surfaces),
                time.time() - total_start_time,
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


if __name__ == "__main__":
    tensors = torch.rand(10, 4, 4)

    print(tensors)

    content_loss = NGramGraphContentLoss(surfaces=tensors)

    print(max([content_loss(row.reshape(-1)) for row in tensors]))

    print(content_loss(torch.rand(4, 4).reshape(-1)))

    tensors = torch.rand(10, 4, 4)

    print(tensors)

    content_loss = ArrayGraph2DContentLoss(surfaces=tensors)

    print(max([content_loss(tensors[i]) for i in range(tensors.shape[0])]))

    print(content_loss(torch.rand(4, 4)))

    tensors = torch.rand(10, 4, 4)

    print(tensors)

    content_loss = HPG2DContentLoss(surfaces=tensors)

    print(max([content_loss(tensors[i]) for i in range(tensors.shape[0])]))

    print(content_loss(torch.rand(4, 4)))
