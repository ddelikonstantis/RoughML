import concurrent
import cProfile
import io
import itertools
import logging
import os
import pstats
import statistics
import time
from abc import abstractmethod
from functools import wraps
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy.io as sio
from pyinsect.collector.NGramGraphCollector import (
    HPG2DCollector,
    HPG2DCollectorParallel,
)
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger()


class NanoroughSurfaceDataset(object):
    """A dataset of pre-generated nanorough surfaces"""

    def __init__(self, surfaces, subsampling_factor=4, transforms=[]):
        self.surfaces = np.array(surfaces)

        self.subsampling_factor = subsampling_factor
        self.subsampling_value = int(surfaces[0].shape[1] / subsampling_factor)

        self.transforms = transforms

        for transform in self.transforms:
            if hasattr(transform, "callback"):
                transform.callback(self)

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx):
        surface = self.surfaces[idx]

        for transform in self.transforms:
            surface = transform(surface)

        return surface


class NanoroughSurfaceMatLabDataset(NanoroughSurfaceDataset):
    """A dataset of pre-generated nanorough surfaces in `.mat` format"""

    def __init__(
        self,
        surface_dir,
        subsampling_factor=4,
        variable_name="data",
        transforms=[],
        limit=None,
    ):
        assert surface_dir.is_dir(), "%s does not exist or is not a dictionary" % (
            surface_dir,
        )

        surfaces = []
        for file in itertools.islice(surface_dir.iterdir(), limit):
            if file.is_dir() or file.suffix != ".mat":
                continue

        surfaces.append(self.from_matlab(file, variable_name))

        super().__init__(
            surfaces, subsampling_factor=subsampling_factor, transforms=transforms
        )

    @classmethod
    def from_matlab(cls, path_to_mat, variable_name):
        matlab_array = sio.loadmat(str(path_to_mat))
        numpy_array = matlab_array[variable_name]

        return numpy_array


class Configuration(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = Configuration(**value)

            setattr(self, key, value)

    def to_dict(self):
        rv = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configuration):
                value = value.to_dict()

        rv[key] = value

        return rv

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return f"<{self.__class__.__name__} '{str(self)}'>"


class KBinsDiscretizerQuantizer(Configuration):
    def __init__(self, surfaces=None, **kwargs):
        if "encode" not in kwargs:
            kwargs["encode"] = "ordinal"

        self.underlying = KBinsDiscretizer(**kwargs)

        self.original_shape = surfaces.shape[1:]

        self.surfaces = self.underlying.fit_transform(
            surfaces.reshape(surfaces.shape[0], -1)
        )
        self.surfaces = self.surfaces.reshape(*surfaces.shape)

    def __call__(self, tensor):
        return self.underlying.transform(tensor.reshape(1, -1)).reshape(
            *self.original_shape
        )

    def __str__(self):
        return str({"underlying": self.underlying, "shape": self.surfaces.shape})


class ContentLoss(Configuration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = KBinsDiscretizerQuantizer(**kwargs)

        self.surfaces = self.quantizer.surfaces

    @abstractmethod
    def __call__(self, surface):
        return self.quantizer(surface)


def per_row(method=None, *, expected_ndim=2):
    def wrapper(method):
        @wraps(method)
        def wrapper_wrapper(self, matrix, *args, **kwargs):
            if len(matrix.shape) > expected_ndim:
                return np.array([method(self, row, *args, **kwargs) for row in matrix])

            return method(self, matrix, *args, **kwargs)

        return wrapper_wrapper

    return wrapper if method is None else wrapper(method)


class HPG2DContentLoss(ContentLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._build_collector()

    def _build_collector(self):
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
            sum(elapsed_time),
            statistics.mean(elapsed_time),
            statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
        )

    def __len__(self):
        return len(self.surfaces)

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))

    def __str__(self):
        return str({"shape": self.surfaces.shape})


class HPG2DContentLossParallel(HPG2DContentLoss):
    def _build_collector(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
            self._collector = HPG2DCollectorParallel(pool=pool)

            logger.info("Submitting %02d jobs", len(self.surfaces))

            futures = {}
            for index, surface in enumerate(self.surfaces):
                future = pool.submit(self._collector._construct_graph, surface)

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
                sum(elapsed_time),
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


class Profiler(object):
    def __init__(self, csv_path=Path.cwd() / "profiler.csv"):
        self._csv_path = csv_path
        self._profiler = cProfile.Profile()

    def __enter__(self):
        self._profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._profiler.disable()

        result = io.StringIO()
        pstats.Stats(self._profiler, stream=result).print_stats()
        result = result.getvalue()

        result = "ncalls" + result.split("ncalls")[-1]
        result = "\n".join(
            [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
        )

        with self._csv_path.open("w+") as csv:
            csv.write(result)

        pd.read_csv(self._csv_path).sort_values(
            by=["tottime", "cumtime"], ascending=False
        ).to_csv(
            self._csv_path, index=False
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        level=logging.INFO,
        # filename=Path.cwd() / f"{__file__}.log",
    )

    # DATASET_PATH = (
    #     Path("/") / "mnt" / "g" / "My Drive" / "Thesis" / "Datasets" / "surfaces.zip"
    # )

    # DATASET_SIZE = 1

    # if DATASET_PATH.is_file():
    #     SURFACES_DIR = Path.cwd() / "surfaces"

    #     if not SURFACES_DIR.is_dir():
    #         SURFACES_DIR.mkdir(parents=True, exist_ok=True)

    #         with ZipFile(DATASET_PATH, "r") as zip_file:
    #             zip_file.extractall(SURFACES_DIR)

    # start_time = time.time()
    # dataset = NanoroughSurfaceMatLabDataset(SURFACES_DIR, limit=DATASET_SIZE)
    # logger.info("NanoroughSurfaceMatLabDataset took %07.3fs" % (time.time() - start_time,))

    start_time = time.time()

    with Profiler():
        content_loss = HPG2DContentLoss(surfaces=np.random.random((2, 16, 16)) * 255)

    logger.info("HPG2DContentLoss took %07.3fs" % (time.time() - start_time,))
