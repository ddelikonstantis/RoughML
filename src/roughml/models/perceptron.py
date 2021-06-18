import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class PerceptronGenerator(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.float64):
        super().__init__()

        self.in_features, self.out_features = in_features, out_features

        self.feature_dims = (in_features,)

        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

        self.to(dtype=dtype)

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        batch = self.activation(self.linear(batch))

        logger.debug("%s:Output: %s", self.__class__.__name__, batch.shape)

        return batch

    @classmethod
    def from_device(
        cls, device, in_features=100, out_features=128 * 128, dtype=torch.float64
    ):
        model = cls(in_features, out_features, dtype=dtype)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)

        return model

    @classmethod
    def from_dataset(cls, dataset, device, in_features=100, dtype=torch.float64):
        out_features = (dataset.subsampling_factor * dataset.subsampling_value) ** 2

        return cls.from_device(
            device, in_features=in_features, out_features=out_features, dtype=dtype
        )


class PerceptronDiscriminator(nn.Module):
    def __init__(self, in_features, dtype=torch.float64):
        super().__init__()

        self.feature_dims = (in_features,)

        self.linear = nn.Linear(in_features, 1)
        self.activation = nn.Sigmoid()

        self.to(dtype=dtype)

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        batch = self.activation(self.linear(batch))

        logger.debug("%s:Output: %s", self.__class__.__name__, batch.shape)

        return batch

    @classmethod
    def from_generator(cls, generator, dtype=torch.float64, device=None):
        model = cls(generator.out_features, dtype=dtype)

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)

        return model
