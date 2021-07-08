import itertools
import logging
from zipfile import ZipFile

import scipy.io as sio

from roughml.data.generators import NonGaussianSurfaceGenerator
from roughml.data.sets import NanoroughSurfaceDataset

logger = logging.getLogger(__name__)


def load_dataset_from_generator(transforms=[], limit=None, **kwargs):
    generate_surfaces = NonGaussianSurfaceGenerator(**kwargs)

    surfaces = []
    for surface in generate_surfaces(limit):
        surfaces.append(surface)

    yield None, NanoroughSurfaceDataset.from_list(surfaces, transforms=transforms)


def load_dataset_from_zip(dataset_path, transforms=[], limit=None, cache_dir=None):
    if not cache_dir.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(dataset_path, "r") as zip_file:
            zip_file.extractall(cache_dir)

    surfaces = []
    for file in itertools.islice(cache_dir.iterdir(), limit):
        if file.is_dir() or file.suffix != ".mat":
            continue

        matlab_array = sio.loadmat(file)
        numpy_array = matlab_array["data"]

        surfaces.append(numpy_array)

    yield dataset_path, NanoroughSurfaceDataset.from_list(
        surfaces, transforms=transforms
    )


def load_dataset_from_pt(dataset_path, transforms=[], limit=None):
    dataset = NanoroughSurfaceDataset.from_pt(dataset_path)

    if limit is not None:
        dataset.surfaces = dataset.surfaces[:limit, :]

    if transforms:
        dataset.transforms = dataset.transforms

    yield dataset_path, dataset


def load_multiple_datasets_from_pt(datasets_dir, transforms=[], limit=None):
    for file in itertools.islice(datasets_dir.iterdir(), limit):
        if file.is_dir() or file.suffix != ".pt":
            continue

        yield next(load_dataset_from_pt(file, transforms=transforms, limit=limit))
