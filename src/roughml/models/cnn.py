import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CNNGenerator(nn.Module):
    def __init__(
        self,
        in_channels=100,
        out_channels=128,
        training_channels=1,
        dtype=torch.float64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training_channels = training_channels

        self.feature_dims = (in_channels, 1, 1)

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels * 16, 4, 1, 0, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 16),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 16, out_channels * 8, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 8),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 8, out_channels * 4, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 4, out_channels * 2, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 2, out_channels, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels, training_channels, 4, 2, 1, bias=False
                    ),
                    nn.ReLU(),
                ),
            ]
        )

        self.to(dtype)

    @classmethod
    def from_device(cls, device):
        model = cls()

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)

        return model

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        for i, module in enumerate(self.module_list):
            batch = module(batch)

            logger.debug(
                "%s:%s #%02d: %s",
                self.__class__.__name__,
                module.__class__.__name__,
                i + 1,
                batch.shape,
            )

        return batch


class CNNDiscriminator(nn.Module):
    def __init__(self, out_channels=1, in_channels=128, dtype=torch.float64):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 8, 1, 8, 1, 0, bias=False), nn.Sigmoid()
                ),
            ]
        )

        self.to(dtype=dtype)

    @classmethod
    def from_device(cls, device):
        model = cls()

        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)

        return model

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        for i, module in enumerate(self.module_list):
            batch = module(batch)

            logger.debug(
                "%s:%s #%02d: %s",
                self.__class__.__name__,
                module.__class__.__name__,
                i + 1,
                batch.shape,
            )

        return batch
