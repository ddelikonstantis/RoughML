import itertools
import logging
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
from pyinsect.collector.NGramGraphCollector import HPG2DCollector
from sklearn.preprocessing import KBinsDiscretizer


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
                return torch.tensor(
                    [method(self, row, *args, **kwargs) for row in matrix]
                )

            return method(self, matrix, *args, **kwargs)

        return wrapper_wrapper

    return wrapper if method is None else wrapper(method)


class HPG2DContentLoss(ContentLoss):
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
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
        level=logging.DEBUG,
        filename=Path.cwd() / f'{__file__}.log'
    )

    DATASET_PATH = (
        Path("/") / "mnt" / "g" / "My Drive" / "Thesis" / "Datasets" / "surfaces.zip"
    )

    DATASET_SIZE = 1

    if DATASET_PATH.is_file():
        SURFACES_DIR = Path.cwd() / "surfaces"
        SURFACES_DIR.mkdir(parents=True, exist_ok=True)

        with ZipFile(DATASET_PATH, "r") as zip_file:
            zip_file.extractall(SURFACES_DIR)

    dataset = NanoroughSurfaceMatLabDataset(SURFACES_DIR, limit=DATASET_SIZE)

    content_loss = HPG2DContentLoss(surfaces=dataset.surfaces)
