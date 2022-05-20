import logging

import torch
from tqdm import tqdm

from roughml.shared.configuration import Configuration
from roughml.shared.decorators import benchmark
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

        # if not hasattr(self, "NGramGraphLoss"):
        self.NGramGraphLoss = None

        # if not hasattr(self, "HeightHistogramAndFourierLoss"):
        self.HeightHistogramAndFourierLoss = None

        if self.NGramGraphLoss is None:
            self.content_loss_weight = 0
        else:
            if not hasattr(self.criterion, "weight"):
                self.content_loss_weight = 1
            else:
                self.content_loss_weight = self.criterion.weight

        if self.HeightHistogramAndFourierLoss is None:
            self.HeightHistogramAndFourier_loss_weight = 0
        else:
            if not hasattr(self.criterion, "weight"):
                self.HeightHistogramAndFourier_loss_weight = 1
            else:
                self.HeightHistogramAndFourier_loss_weight = self.criterion.weight

        if not hasattr(self.criterion, "weight"):
            self.criterion.weight = 1

        if self.content_loss_weight != 0 and self.HeightHistogramAndFourier_loss_weight != 0:
            self.criterion.weight = self.criterion.weight / (self.criterion.weight + self.content_loss_weight + self.HeightHistogramAndFourier_loss_weight)
            self.content_loss_weight = self.criterion.weight
            self.HeightHistogramAndFourier_loss_weight = self.criterion.weight
        elif self.content_loss_weight != 0:
            self.criterion.weight = self.criterion.weight / (self.criterion.weight + self.content_loss_weight + self.HeightHistogramAndFourier_loss_weight)
            self.content_loss_weight = self.criterion.weight
        elif self.HeightHistogramAndFourier_loss_weight != 0:
            self.criterion.weight = self.criterion.weight / (self.criterion.weight + self.content_loss_weight + self.HeightHistogramAndFourier_loss_weight)
            self.HeightHistogramAndFourier_loss_weight = self.criterion.weight
        else:
            self.criterion.weight /= self.criterion.weight

        if not hasattr(self, "fixed_noise_dim"):
            self.fixed_noise_dim = 128

    def __call__(self, generator, discriminator, dataset):
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

        min_generator_loss = float("inf")
        max_generator_loss = float(0.0)
        max_discriminator_loss = float(0.0)
        max_HeightHistogramAndFourierLoss = float(0.0)
        max_NGramGraphLoss = float(0.0)
        patience = 20       # number of epochs where generator loss shows no significant change
        loss_change = []
        early_stop = False
        delta = 0.01        # generator loss threshold that shows no significant change
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
            ) = train_epoch_f(
                generator,
                discriminator,
                train_dataloader,
                optimizer_generator,
                optimizer_discriminator,
                self.criterion.instance,
                content_loss_fn=self.NGramGraphLoss,
                vector_content_loss_fn=self.HeightHistogramAndFourierLoss,
                loss_weights=(self.criterion.weight, self.content_loss_weight, self.HeightHistogramAndFourier_loss_weight),
                log_every_n=self.log_every_n,
                load_checkpoint = self.load_checkpoint,
            )

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
                "Epoch: %02d, Generator Loss: %7.3f (N-Gram Graph Loss: %7.3f, Height Histogram and Fourier Loss: %7.3f), Discriminator Loss: %7.3f",
                epoch,
                generator_loss,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
                discriminator_loss,
            )

            # normalize all losses from 0 to 1
            if generator_loss > max_generator_loss:
                max_generator_loss = generator_loss
            generator_loss /= max_generator_loss

            if discriminator_loss > max_discriminator_loss:
                max_discriminator_loss = discriminator_loss
            discriminator_loss /= max_discriminator_loss

            if HeightHistogramAndFourierLoss > max_HeightHistogramAndFourierLoss:
                max_HeightHistogramAndFourierLoss = HeightHistogramAndFourierLoss
            HeightHistogramAndFourierLoss /= max_HeightHistogramAndFourierLoss

            if NGramGraphLoss > max_NGramGraphLoss:
                max_NGramGraphLoss = NGramGraphLoss
            NGramGraphLoss /= max_NGramGraphLoss

            logger.info(
                "Epoch: %02d, Normalized Generator Loss: %7.3f (Normalized N-Gram Graph Loss: %7.3f, Normalized Height Histogram and Fourier Loss: %7.3f), Normalized Discriminator Loss: %7.3f",
                epoch,
                generator_loss,
                NGramGraphLoss,
                HeightHistogramAndFourierLoss,
                discriminator_loss,
            )

            logger.info(
                "Epoch: %02d, Discriminator Output: [Real: %7.3f, Fake: %7.3f]",
                epoch,
                discriminator_output_real,
                discriminator_output_fake,
            )

            # stop the training when generator loss shows no significant change after a consecutive number of epochs
            loss_change.append(generator_loss)
            cntr = 0
            for i in range(1, len(loss_change)):
                if (loss_change[i] > (loss_change[i-1] + delta)) or (loss_change[i] < (loss_change[i-1] - delta)):
                    cntr = 0
                else:
                    cntr += 1

            if cntr >= patience:
                early_stop = True

            if early_stop:
                logger.info(
                "Early stopping in epoch %02d since loss did not improve after %02d epochs",
                i,
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
                fixed_fake,
            )