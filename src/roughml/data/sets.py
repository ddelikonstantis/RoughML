import itertools

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset


class NanoroughSurfaceDataset(Dataset):
    """A dataset of pre-generated nanorough surfaces"""

    def __init__(self, surfaces, subsampling_factor, transforms):
        self.surfaces = surfaces

        self.subsampling_factor = subsampling_factor
        self.subsampling_value = int(self.surfaces.shape[2] / subsampling_factor)

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

    def to_pt(self, path):
        state = {
            "surfaces": self.surfaces,
            "subsampling_factor": self.subsampling_factor,
            "transforms": self.transforms,
        }

        torch.save(state, path)

    @classmethod
    def from_list(cls, surfaces, subsampling_factor=4, transforms=[]):
        surfaces = np.array(surfaces)
        surfaces = torch.from_numpy(surfaces)

        return cls(surfaces, subsampling_factor, transforms)

    @classmethod
    def from_matlab(
        cls,
        surface_dir,
        subsampling_factor=4,
        variable_name="data",
        transforms=[],
        limit=None,
    ):
        """Load pre-generated nanorough surfaces in `.mat` format"""

        assert surface_dir.is_dir(), "%s does not exist or is not a dictionary" % (
            surface_dir,
        )

        surfaces = []
        for file in itertools.islice(surface_dir.iterdir(), limit):
            if file.is_dir() or file.suffix != ".mat":
                continue

            matlab_array = sio.loadmat(file)
            numpy_array = matlab_array[variable_name]

            surfaces.append(numpy_array)

        return cls.from_list(
            surfaces, subsampling_factor=subsampling_factor, transforms=transforms
        )

    @classmethod
    def from_pt(cls, path):
        state = torch.load(path)

        return cls(state["surfaces"], state["subsampling_factor"], state["transforms"])
