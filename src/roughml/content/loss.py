import concurrent
import contextlib
import itertools
import logging
import os
import pickle
import statistics
import time
from abc import ABC, abstractmethod
from functools import wraps
import math

import numpy as np
import scipy.fft as fft
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
    def __init__(self, skip_quantization=False, **kwargs):
        super().__init__(**kwargs)

        self.skip_quantization = skip_quantization

        if self.skip_quantization is False:
            self.quantizer = KBinsDiscretizerQuantizer(**kwargs)
            self.surfaces = self.quantizer.surfaces

    def __len__(self):
        return len(self.surfaces)

    @abstractmethod
    def __call__(self, surface):
        if self.skip_quantization is False:
            return self.quantizer(surface)

        return surface

    def __str__(self):
        return str({"shape": self.surfaces.shape})

    def to_pickle(self, path):
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path):
        with path.open("rb") as file:
            instance = pickle.load(file)

            if not isinstance(instance, cls):
                raise TypeError(
                    "%r is not an instance of %s" % (instance, cls.__name__)
                )

            return instance


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

        logger.debug("Constructing %02d graphs", len(self.surfaces))

        elapsed_time = []
        for index, surface in enumerate(self.surfaces):
            start_time = time.time()
            self._collector.add(surface)
            elapsed_time.append(time.time() - start_time)

            logger.debug("Constructed graph %02d in %07.3fs", index, elapsed_time[-1])

        logger.debug(
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

            logger.debug("Awaiting %02d jobs", len(self.surfaces))

            elapsed_time = [None] * len(self.surfaces)
            for future in concurrent.futures.as_completed(futures):
                self._collector._add_graph(future.result())

                index, start_time = futures[future]

                elapsed_time[index] = time.time() - start_time

                logger.debug(
                    "Job %02d completed after %07.3fs", index, elapsed_time[index]
                )

            logger.debug(
                "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
                len(self.surfaces),
                time.time() - total_start_time,
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


class VectorSpaceContentLoss(ContentLoss):
    """
    A content loss that represents surfaces as vectors.

    The content loss calculates the historgram and the fourier transform
    corresponding to each provided surface, in order to construct a vector
    corresponding to that surface. It then utilizes conventional vector
    distance metrics to calculate the loss value.
    """

    def __init__(self, **kwargs):
        super().__init__(skip_quantization=True, **kwargs)

        with contextlib.suppress(AttributeError):
            self.surfaces = self.surfaces.numpy()

        if not hasattr(self, "n_neighbors"):
            # neighbours is total number of surfaces
            self.n_neighbors = len(self.surfaces)   
        
        # get global min and max height values of all surfaces
        self.HistogramMaxVal, self.HistogramMinVal = float(0.0), float('inf')
        for surface in self.surfaces:
            if np.min(surface) < self.HistogramMinVal:
                self.HistogramMinVal = np.min(surface)
            if np.max(surface) > self.HistogramMaxVal:
                self.HistogramMaxVal = np.max(surface)
        # get second standard deviation of global min and max height values
        # self.HistogramMinVal = self.HistogramMinVal * 2
        # self.HistogramMaxVal = self.HistogramMaxVal * 2

        # histogram bins formula according to feature dimension
        self.bins = max(10, 10**(math.ceil(math.log10(128**2)) - 3))

        self.histograms, self.fouriers = [], []
        for surface in self.surfaces:
            # create histogram of current surface and append to histogram list
            self.histograms.append(np.histogram(surface.reshape(-1), bins=self.bins, range=(self.HistogramMinVal, self.HistogramMaxVal))[0])
            # compute fourier of current surface and append to fourier list
            self.fouriers.append(np.absolute(fft.fft2(surface)))


    # TODO: examine cosine similarity
    @per_row
    # Returns a measurement of loss, based on the difference of (a) height distributions (b) 2D FFT components, between 
    # the input surface and the - pre-provided - subset of training surfaces. The exact subset cardinality is based on the n_neighbors parameter
    # of the class instance.
    def __call__(self, surface):
        # Get (a) the histogram of the heights and (b) the real components of the 2D FFT for the evaluated surface
        #TODO: exception error: too many values to unpack
        (histogram, _), fourier = np.histogram(surface.reshape(-1), bins=self.bins, range=(self.HistogramMinVal, self.HistogramMaxVal))[0], np.absolute(
            fft.fft2(surface)
        )

        # Initialize the difference
        diff = 0
        # For every (histogram of heights , fourier) tuple of each training instance (of an n_neighbors subset of the training set)
        for _histogram, _fourier in itertools.islice(
            zip(self.histograms, self.fouriers), self.n_neighbors
        ):
            # Calculate the histogram contribution to the loss with respect to this training instance
            # This contribution is effectively a difference/dissimilarity measurement between histograms
            #i.e. for every point in the evaluated surface histogram, calculate the difference of its value to the current training instance histogram 
            # and raise this difference to the power of 2.
            # Get the square root of the mean of these squared differences and add it to the total diff
            # TODO: histogram ranges can vary according to height values. Alignment is needed.
            diff += np.sqrt(np.square(np.subtract(histogram, _histogram)).mean()) 

            # Calculate the Fourier contribution to the loss with respect to this training instance
            # This contribution is effectively a difference/dissimilarity measurement between Fourier transformation outputs
            #i.e. for every point in the fourier transform of the evaluated surface, calculate the difference of its value to the 
            # corresponding fourier transform point of the current training instance
            # and raise the difference to the power of 2.
            # Get the square root of the mean of these squared differences and add it to the total diff
            # TODO: Can the semantics of fft components vary according to spectral content? Is alignment needed?
            diff += np.sqrt(np.square(np.subtract(fourier, _fourier)).mean())
        
        # Normalize the total diff to reflect the average diff over all instances (each of which has 2 components contributing: histogram and Fourier)
        diff /= self.n_neighbors * 2
        
        # The loss is a function of diff normalized between 0 and 1. A value of diff = 0
        # (i.e. the evaluated surface is completely identical - in terms of histogam and Fourier - to all the training instances)
        # will give a max loss of 1. Higher values of diff will result to lower loss values (to the limit reaching zero).
        loss = 1 - (1 / (1 + diff))
        
        # Return the normalized loss
        return loss


if __name__ == "__main__":
    SIZE, DIM = 10, 4

    fixed_noise, tensors = torch.rand(DIM, DIM), torch.rand(SIZE, DIM, DIM)
    print(fixed_noise, '\n', tensors)

    print("\nTesting 'NGramGraphContentLoss'")
    content_loss = NGramGraphContentLoss(surfaces=tensors)

    content_losses = [content_loss(row.reshape(-1)) for row in tensors]

    print(
        content_loss(fixed_noise.reshape(-1)),
        (min(content_losses), max(content_losses)),
    )

    print("\nTesting 'ArrayGraph2DContentLoss'")
    content_loss = ArrayGraph2DContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i]) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise), (min(content_losses), max(content_losses)))

    print("\nTesting 'HPG2DContentLoss'")
    content_loss = HPG2DContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i]) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise), (min(content_losses), max(content_losses)))

    print("\nTesting 'VectorSpaceContentLoss'")
    content_loss = VectorSpaceContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i].numpy()) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise.numpy()), (min(content_losses), max(content_losses)))