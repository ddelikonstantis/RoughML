import concurrent
import cProfile
import datetime
import enum
import io
import itertools
import logging
import os
import pickle
import pstats
import re
import statistics
import time
from abc import abstractmethod
from functools import wraps
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import typer
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


class HPG2DContentLossParallelAddition(HPG2DContentLoss):
    def _build_collector(self):
        logger.info("Using an addition parallel content loss")

        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
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
                sum(elapsed_time),
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


class HPG2DContentLossParallelConstruction(HPG2DContentLoss):
    def _build_collector(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
            self._collector = HPG2DCollectorParallel(pool=pool)

            logger.info("Constructing %02d graphs", len(self.surfaces))

            elapsed_time = []
            for index, surface in enumerate(self.surfaces):
                start_time = time.time()
                self._collector.add(surface)
                elapsed_time.append(time.time() - start_time)

                logger.info(
                    "Constructed graph %02d in %07.3fs", index, elapsed_time[-1]
                )

            logger.info(
                "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
                len(self.surfaces),
                sum(elapsed_time),
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


class HPG2DContentLossParallel(HPG2DContentLoss):
    def _build_collector(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
            self._collector = HPG2DCollectorParallel(pool=pool)

            logger.info("Constructing %02d graphs", len(self.surfaces))

            futures = {}
            for index, surface in enumerate(self.surfaces):
                future = pool.submit(self._collector.add, surface)

                futures[future] = (index, time.time())

            logger.info("Awaiting %02d jobs", len(self.surfaces))

            elapsed_time = [None] * len(self.surfaces)
            for future in concurrent.futures.as_completed(futures):
                future.result()

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


def unique_name_of(prefix, extension="csv"):
    strtime = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S_%f")

    return f"{prefix}_{strtime}.{extension}"


def to_snake_case(name):
    if "___" in name:
        return name.lower()

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = name.replace("__", "_")

    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def unique_cwd_of(prefix, extension="csv"):
    return Path.cwd() / unique_name_of(to_snake_case(prefix), extension=extension)


class Profiler(object):
    def __init__(self, csv_path=unique_cwd_of("profiler")):
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
        ).to_csv(self._csv_path, index=False)


class LoggingLevel(enum.Enum):
    info = "info"
    debug = "debug"
    critical = "critical"


def main(
    dims: Tuple[int, int, int] = typer.Argument((1, 16, 16)),
    logging_lvl: LoggingLevel = "info",
    log_to_file: bool = False,
    parallel: int = 0,
):
    logging_lvls = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "critical": logging.CRITICAL,
    }

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        level=logging_lvls[logging_lvl.value],
        filename=Path.cwd() / f"{__file__}_{str(datetime.datetime.now())}.log"
        if log_to_file
        else None,
    )

    start_time = time.time()

    contnet_losses = [
        HPG2DContentLoss,
        HPG2DContentLossParallelConstruction,
        HPG2DContentLossParallelAddition,
        HPG2DContentLossParallel,
    ]

    with Profiler(unique_cwd_of(contnet_losses[parallel].__name__)):
        content_loss = contnet_losses[parallel](surfaces=(np.random.random(dims) * 255))

        logger.info("Pickling an instance of %s", content_loss.__class__.__name__)
        with unique_cwd_of(content_loss.__class__.__name__, extension="pkl").open(
            "wb"
        ) as pkl:
            pickle.dump(content_loss, pkl)

    logger.info(
        "%s took %07.3fs"
        % (
            contnet_losses[parallel].__name__,
            time.time() - start_time,
        )
    )


if __name__ == "__main__":
    typer.run(main)
