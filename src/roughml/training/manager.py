import logging

import torch
from tqdm import tqdm

from roughml.shared.configuration import Configuration
from roughml.shared.decorators import benchmark
from roughml.shared.early_stop import early_stopping
from roughml.training.split import train_test_dataloaders

logger = logging.getLogger(__name__)


class TrainingManager(Configuration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not hasattr(self, "benchmark"):
            self.benchmark = False

        if not hasattr(self, "log_every_n"):
            self.log_every_n = None

        if not hasattr(self, "checkpoint"):
            self.checkpoint = Configuration(directory=None, multiple=False)

        # check if N-Gram Graph loss is taken into account or not
        if not hasattr(self, "NGramGraphLoss"):
            self.NGramGraphLoss = None

        # check if Height Histogram and Fourier loss is taken into account or not
        if not hasattr(self, "HeightHistogramAndFourierLoss"):
            self.HeightHistogramAndFourierLoss = None

        # Initialize the N-Gram Graph loss weight
        if self.NGramGraphLoss is None:
            self.content_loss_weight = 0
        else:
            if not hasattr(self.criterion, "weight"):
                self.content_loss_weight = 1
            else:
                # make equal to bce weight
                self.content_loss_weight = self.criterion.weight

        # Initialize the Height Histogram and Fourier loss weight
        if self.HeightHistogramAndFourierLoss is None:
            self.HeightHistogramAndFourier_loss_weight = 0
        else:
            if not hasattr(self.criterion, "weight"):
                self.HeightHistogramAndFourier_loss_weight = 1
            else:
                # make equal to bce weight
                self.HeightHistogramAndFourier_loss_weight = self.criterion.weight

        # Initialize the weight
        if not hasattr(self.criterion, "weight"):
            self.criterion.weight = 1
        # Calculate the normalization factor
        norm_factor = (self.criterion.weight + self.content_loss_weight + self.HeightHistogramAndFourier_loss_weight)
        # if the factor is zero, raise an exception
        if norm_factor == 0:
            raise RuntimeError("All weights related to the losses are zero. Please revise.")
        # Otherwise, we normalize as expected
        self.criterion.weight = self.criterion.weight / norm_factor
        self.content_loss_weight = self.content_loss_weight / norm_factor
        self.HeightHistogramAndFourier_loss_weight = self.HeightHistogramAndFourier_loss_weight / norm_factor

        # Initialize fixed_noise_dim
        if not hasattr(self, "fixed_noise_dim"):
            self.fixed_noise_dim = 64

    def __call__(self, generator, discriminator, dataset):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = torch.randn(
            self.fixed_noise_dim,
            *generator.feature_dims,
            dtype=generator.dtype,
            device=generator.device,
        )

        train_dataloader, _ = train_test_dataloaders(
            dataset, train_ratio=self.train_ratio, **self.dataloader.to_dict()
        )

        optimizer_generator = self.optimizer.type(
            generator.parameters(), **self.optimizer.params.to_dict()
        )

        optimizer_discriminator = self.optimizer.type(
            discriminator.parameters(), **self.optimizer.params.to_dict()
        )

        train_epoch_f = self.train_epoch

        if self.benchmark is True and logger.level <= logging.INFO:
            train_epoch_f = benchmark(train_epoch_f)

        # set minimum generator loss var to compare with actual generator loss
        # in every epoch in order to save the best model as a checkpoint
        min_generator_loss = float("inf")

        # set maximum variables for every loss normalization per batch
        max_dis_bce_loss_real, max_dis_bce_loss_fake = 0, 0
        max_gen_bce_loss, max_gen_NGramGraphLoss, max_gen_HeightHistogramAndFourierLoss = 0, 0, 0
        maximum_losses = {'max_gen_bce_loss': max_gen_bce_loss,
                          'max_gen_NGramGraphLoss': max_gen_NGramGraphLoss,
                          'max_gen_HeightHistogramAndFourierLoss': max_gen_HeightHistogramAndFourierLoss,
                          'max_dis_bce_loss_real': max_dis_bce_loss_real,
                          'max_dis_bce_loss_fake': max_dis_bce_loss_fake,
                          }

        # variables for storing the raw losses per epoch to be viewed in the log file
        raw_dis_bce_loss_real, raw_dis_bce_loss_fake = 0, 0
        raw_gen_bce_loss, raw_gen_NGramGraphLoss, raw_gen_HeightHistogramAndFourierLoss = 0, 0, 0
        raw_losses =    {'raw_gen_bce_loss': raw_gen_bce_loss,
                         'raw_gen_NGramGraphLoss': raw_gen_NGramGraphLoss,
                         'raw_gen_HeightHistogramAndFourierLoss': raw_gen_HeightHistogramAndFourierLoss,
                         'raw_dis_bce_loss_real': raw_dis_bce_loss_real,
                         'raw_dis_bce_loss_fake': raw_dis_bce_loss_fake,
                        }

        # set limits for early stopping
        # early stopping occurs when generator loss does not change significantly
        # for example we want to stop the training when its stuck in a local minimum
        patience = 10       # number of consecutive epochs where generator loss shows no significant change
        delta = 0.001       # generator loss change threshold
        gen_loss_hist = []  # initialize list for saving past generator losses
        early_stop = False

        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
                BCELoss,
                losses_max,
                losses_raw_val,
            ) = train_epoch_f(
                generator,
                discriminator,
                train_dataloader,
                optimizer_generator,
                optimizer_discriminator,
                self.criterion.instance,
                content_loss_fn=self.NGramGraphLoss,
                vector_content_loss_fn=self.HeightHistogramAndFourierLoss,
                loss_weights=[self.criterion.weight, self.content_loss_weight, self.HeightHistogramAndFourier_loss_weight],
                losses_maxima=maximum_losses,
                losses_raw=raw_losses,
                log_every_n=self.log_every_n,
                load_checkpoint = self.load_checkpoint,
            )

            # update maximum losses seen in current epoch
            maximum_losses = losses_max

            if (
                self.checkpoint.directory is not None
                and generator_loss < min_generator_loss
            ):
                min_generator_loss = generator_loss

                generator_mt, discriminator_mt = (
                    f"{generator.__class__.__name__}",
                    f"{discriminator.__class__.__name__}",
                )

                if self.checkpoint.multiple is True:
                    generator_mt += f"_{epoch:03d}"
                    discriminator_mt += f"_{epoch:03d}"

                torch.save(
                    generator.state_dict(),
                    self.checkpoint.directory / f"{generator_mt}.pt",
                )

                torch.save(
                    discriminator.state_dict(),
                    self.checkpoint.directory / f"{discriminator_mt}.pt",
                )

            with torch.no_grad():
                fixed_fake = generator(fixed_noise).detach().cpu()

            logger.info(
                "Epoch:%02d, Raw Gen BCE Loss:%7.3f, Raw Gen NGG Loss:%7.3f, Raw Gen HFF Loss:%7.3f, Raw Dis BCE Loss Real:%7.3f, Raw Dis BCE Loss Fake:%7.3f", 
                epoch,
                losses_raw_val['raw_gen_bce_loss'],
                losses_raw_val['raw_gen_NGramGraphLoss'],
                losses_raw_val['raw_gen_HeightHistogramAndFourierLoss'],
                losses_raw_val['raw_dis_bce_loss_real'],
                losses_raw_val['raw_dis_bce_loss_fake'],
            )

            logger.info(
                "Epoch:%02d, Norm Total Generator Loss:%7.5f, Norm BCE Loss:%7.5f, Norm NGG Loss:%7.5f, Norm HFF Loss:%7.5f, Norm Total Discriminator Loss:%7.5f",
                epoch,
                generator_loss,
                BCELoss,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
                discriminator_loss,
            )

            logger.info(
                "Epoch:%02d, Discriminator Output: [Real images:%7.5f, Generated images:%7.5f]",
                epoch,
                discriminator_output_real,
                discriminator_output_fake,
            )

            # stop the training procedure when generator loss shows no significant change 
            # after a predefined consecutive number of epochs
            gen_loss_hist.append(generator_loss)  # save the generator losses so far
            early_stop, index = early_stopping(gen_loss_hist, patience, delta)

            if early_stop:
                logger.info(
                "Stopping early the training in epoch %02d since loss did not improve in previous %02d consecutive epochs",
                index,
                patience
                )

                break


            yield (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
                BCELoss,
                fixed_fake,
            )