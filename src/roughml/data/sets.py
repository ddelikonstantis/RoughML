import itertools

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.dataset import Dataset


class NanoroughSurfaceDataset(Dataset):
    """A dataset of pre-generated nanorough surfaces"""

    def __init__(self, surfaces, subsampling_factor=4, transforms=[]):
        self.surfaces = np.array(surfaces)
        self.surfaces = torch.from_numpy(self.surfaces)

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
