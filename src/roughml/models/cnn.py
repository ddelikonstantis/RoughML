import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

from roughml.models.base import Base


class CNNBase(Base):
    @classmethod
    # custom weights initialization called on netG and netD
    # all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. 
    # The initialize_weights function takes an initialized model as input and reinitializes all convolutional, 
    # convolutional-transpose, and batch normalization layers to meet this criteria. 
    # This function is applied to the models immediately after initialization.
    def initialize_weights(cls, model):
        classname = model.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    @classmethod
    def from_device(
        cls, device, *args, dtype=torch.float64, gradient_clipping=None, **kwargs
    ):
        model = super().from_device(
            device, *args, dtype=dtype, gradient_clipping=gradient_clipping, **kwargs
        )

        # Apply the initialize_weights function to randomly initialize all weights
        # to mean=0, stdev=0.02.
        model.apply(cls.initialize_weights)

        return model


class CNNGenerator(Base):
    def __init__(
        self,
        in_channels=128,        # Size of z latent vector (i.e. size of generator input)
        out_channels=128,       # Size of feature maps in generator
        training_channels=1,    # Number of channels in the training images. For color images this is 3. For grayscale this is 1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training_channels = training_channels

        self.feature_dims = (in_channels, 1, 1)

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    # input is in_channels (latent vector Z), going into a convolution
                    nn.ConvTranspose2d(
                        in_channels, out_channels * 16, 4, 1, 0, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 16),
                    nn.ReLU(True),
                ), # state size. (out_channels*16) x 4 x 4
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 16, out_channels * 8, 4, 1, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 8),
                    nn.ReLU(True),
                ), # state size. (out_channels*8) x 8 x 8
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 8, out_channels * 4, 4, 1, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.ReLU(True),
                ), # state size. (out_channels*4) x 16 x 16
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 4, out_channels * 2, 6, 2, 1, bias=False
                    ), # state size. (out_channels*2) x 32 x 32
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels * 2, out_channels, 6, 3, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ), # state size. (out_channels) x 64 x 64
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels, training_channels, 6, 3, 2, bias=False
                    ), # state size. (training_channels) x 128 x 128
                    # nn.Tanh(),    # removed to avoid normalization [-1, 1] of the height of surfaces
                ),
            ]
        )

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


class CNNDiscriminator(Base):
    def __init__(self, out_channels=1, in_channels=128):
        super().__init__()

        self.out_channels = out_channels    # Number of channels in the training images. For color images this is 3. For grayscale this is 1
        self.in_channels = in_channels      # Size of feature maps in discriminator

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, in_channels, 4, 4, 4, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ), # state size. (in_channels) x 64 x 64
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 2, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ), # state size. (in_channels*2) x 32 x 32
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels * 4, 2, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                ), # state size. (in_channels*4) x 16 x 16
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels * 8, 2, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                ), # state size. (in_channels*8) x 8 x 8
                nn.Sequential(
                    nn.Conv2d(in_channels * 8, in_channels * 16, 2, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 16),
                    nn.LeakyReLU(0.2, inplace=True),
                ), # state size. (in_channels*16) x 4 x 4
                nn.Sequential(
                    nn.Conv2d(in_channels * 16, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                ), # state size. 1
            ]
        )

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

    @classmethod
    def from_generator(cls, generator, dtype=torch.float64, gradient_clipping=None):
        return cls.from_device(
            generator.device,
            dtype=dtype,
            gradient_clipping=gradient_clipping,
        )
