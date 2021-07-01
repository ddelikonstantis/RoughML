import logging
import time

import torch
from torch.optim import Adam
from tqdm import tqdm

from roughml.shared.configuration import Configuration
from roughml.shared.decorators import benchmark
from roughml.training.split import train_test_dataloaders

logger = logging.getLogger(__name__)


def per_epoch(
    generator,
    discriminator,
    dataloader,
    optimizer_generator,
    optimizer_discriminator,
    criterion,
    content_loss=None,
    loss_weights=None,
    log_every_n=None,
):
    generator.train()

    if content_loss is None:
        content_loss_weight, criterion_weight = 0, 1
    else:
        content_loss_weight, criterion_weight = loss_weights

    generator_loss, discriminator_loss = 0, 0
    discriminator_output_real, discriminator_output_fake = 0, 0

    start_time = time.time()
    for train_iteration, X_batch in enumerate(dataloader):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        label = torch.full(
            (X_batch.size(0),), 1, dtype=X_batch.dtype, device=X_batch.device
        )
        # Forward pass real batch through D
        output = discriminator(X_batch).view(-1)
        # Calculate loss on all-real batch
        discriminator_error_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        discriminator_error_real.backward()
        discriminator_output_real_batch = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(
            X_batch.size(0),
            *generator.feature_dims,
            dtype=X_batch.dtype,
            device=X_batch.device,
        )
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        discriminator_error_fake = criterion(output, label)
        # Calculate the gradients for this batch
        discriminator_error_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        discriminator_error_total = discriminator_error_real + discriminator_error_fake
        # Update D
        optimizer_discriminator.step()

        # (2) Update G network: maximize log(D(G(z)))
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        if content_loss_weight <= 0:
            discriminator_error_fake = criterion(output, label)
        else:
            generator_content_loss = content_loss(fake.cpu().detach().numpy().squeeze())
            generator_content_loss = torch.mean(generator_content_loss).to(fake.device)

            discriminator_error_fake = (
                content_loss_weight * generator_content_loss
                + criterion_weight * criterion(output, label)
            )
        # Calculate gradients for G, which propagate through the discriminator
        discriminator_error_fake.backward()
        discriminator_output_fake_batch = output.mean().item()
        # Update G
        optimizer_generator.step()

        generator_loss += discriminator_error_fake.item() / len(dataloader)
        discriminator_loss += discriminator_error_total.item() / len(dataloader)
        discriminator_output_real += discriminator_output_real_batch / len(dataloader)
        discriminator_output_fake += discriminator_output_fake_batch / len(dataloader)

        if log_every_n is not None and not train_iteration % log_every_n:
            logger.info(
                "Training Iteration #%04d ended after %7.3f seconds",
                train_iteration,
                time.time() - start_time,
            )

            start_time = time.time()

    return (
        generator_loss,
        discriminator_loss,
        discriminator_output_real,
        discriminator_output_fake,
    )


class TrainingManager(Configuration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not hasattr(self, "benchmark"):
            self.benchmark = False

        if not hasattr(self, "log_every_n"):
            self.log_every_n = None

        if not hasattr(self, "checkpoint"):
            self.checkpoint = Configuration(directory=None, multiple=False)

        if not hasattr(self, "content_loss"):
            self.content_loss = None

        if isinstance(self.criterion, tuple):
            self.criterion, self.criterion_weight = self.criterion
        else:
            self.criterion_weight = 0.5

        if isinstance(self.content_loss, tuple):
            self.content_loss, self.content_loss_weight = self.content_loss
        else:
            self.content_loss_weight = 1 - self.criterion_weight

        if not hasattr(self, "fixed_noise_dim"):
            self.fixed_noise_dim = 64

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

        optimizer_generator = Adam(generator.parameters(), **self.optimizer.to_dict())

        optimizer_discriminator = Adam(
            discriminator.parameters(), **self.optimizer.to_dict()
        )

        train_epoch_f = self.train_epoch

        if self.benchmark is True and logger.level <= logging.INFO:
            train_epoch_f = benchmark(train_epoch_f)

        min_generator_loss = float("inf")
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
            ) = train_epoch_f(
                generator,
                discriminator,
                train_dataloader,
                optimizer_generator,
                optimizer_discriminator,
                self.criterion,
                content_loss=self.content_loss,
                loss_weights=(self.content_loss_weight, self.criterion_weight),
                log_every_n=self.log_every_n,
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
                    self.checkpoint.directory / f"{generator_mt}.mt",
                )

                torch.save(
                    discriminator.state_dict(),
                    self.checkpoint.directory / f"{discriminator_mt}.mt",
                )

            with torch.no_grad():
                fixed_fake = generator(fixed_noise).detach().cpu()

            logger.info(
                "Epoch: %02d, Generator Loss: %7.3f, Discriminator Loss: %7.3f",
                epoch,
                generator_loss,
                discriminator_loss,
            )

            logger.info(
                "Epoch: %02d, Discriminator Output: [Real: %7.3f, Fake: %7.3f]",
                epoch,
                discriminator_output_real,
                discriminator_output_fake,
            )

            yield (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                fixed_fake,
            )
