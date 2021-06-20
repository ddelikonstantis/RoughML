from abc import abstractmethod
from functools import wraps

import torch
from pyinsect.collector.NGramGraphCollector import (
    ArrayGraph2DCollector,
    HPG2DCollector,
    NGramGraphCollector,
)

from roughml.content.quantization import KBinsDiscretizerQuantizer
from roughml.shared.configuration import Configuration


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


class ContentLoss(Configuration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = KBinsDiscretizerQuantizer(**kwargs)

        self.surfaces = self.quantizer.surfaces

    @abstractmethod
    def __call__(self, surface):
        return self.quantizer(surface)


class NGramGraphContentLoss(ContentLoss):
    """An `n-gram graph` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.surfaces = self.surfaces.reshape(self.surfaces.shape[0], -1)

        self._collector = NGramGraphCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    def __len__(self):
        return len(self.surfaces)

    @per_row(expected_ndim=1)
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface).reshape(-1))

    def __str__(self):
        return str({"shape": self.surfaces.shape})


class ArrayGraph2DContentLoss(ContentLoss):
    """A `2D array graph` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._collector = ArrayGraph2DCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    def __len__(self):
        return len(self.surfaces)

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))

    def __str__(self):
        return str({"shape": self.surfaces.shape})


class HPG2DContentLoss(ContentLoss):
    """A `Hierarchical Proximity Graph (HPG)` based content loss"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._collector = HPG2DCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    def __len__(self):
        return len(self.surfaces)

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))

    def __str__(self):
        return str({"shape": self.surfaces.shape})


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
