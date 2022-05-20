import logging
import time

import torch

from roughml.content.loss import NGramGraphContentLoss

logger = logging.getLogger(__name__)

generator_loss_max = -1.0
NGramGraphContentLoss_max = -1
NGramGraphContentLoss_max = -1

# TODO: Add method description/documentation
# TODO: debug ONLY this function/script
def per_epoch(
    generator,
    discriminator,
    dataloader,
    optimizer_generator,
    optimizer_discriminator,
    criterion,
    content_loss_fn=None,
    vector_content_loss_fn=None,
    loss_weights=[1.0, 1.0, 1.0],
    loss_maxima=[0.0, 0.0, 0.0],
    log_every_n=None,
    load_checkpoint = None,
):
    generator.train()

    generator_loss, discriminator_loss = 0, 0
    discriminator_output_real, discriminator_output_fake = 0, 0
    NGramGraphLoss, HeightHistogramAndFourierLoss, BCELoss = 0, 0, 0

    start_time = time.time()
    for train_iteration, X_batch in enumerate(dataloader):
        
        # change batch type to match model's checkpoint weights when model is loaded
        if load_checkpoint:
            X_batch = X_batch.float()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        label = torch.full(
            (X_batch.size(0),), 1, dtype=X_batch.dtype, device=X_batch.device
        )
        # Forward pass real batch through D (Discriminator)
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
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        discriminator_error_fake.backward()
        # Compute error of D as sum over the fake and the real batches
        discriminator_error_total = discriminator_error_real + discriminator_error_fake
        # Update D
        optimizer_discriminator.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)

        # Calculate G's loss based on this condition
        
        # Initialize overall loss to zero
        overall_loss = 0.0
        # For each loss component        
            # Normalize it based on the maximum value we have seen in this loss
            # loss_maxima[0] is Binary Cross-Entropy maximum so far
            # loss_maxima[1] is NGramGraphLoss  maximum so far
            # loss_maxima[2] is HeightHistogramAndFourierLoss  maximum so far
            # loss_weights[0] is Binary Cross-Entropy weight
            # loss_weights[1] is NGramGraphLoss weight
            # loss_weights[2] is HeightHistogramAndFourierLoss weight
            # Weight it and add it to the overall loss

        # Returns a normalized and a normalized weighted value of a measure. Normalization occurs before weighting.
        # The function returns both the above value, but also the new maxValueSoFar.
        def normalizedAndWeightedLoss(value, maxValueSoFar, weight):
            # Update maximum
            if maxValueSoFar < value:
                maxValueSoFar = value
            # Return normalized value and normalized weighted value, and max value so far
            return (value / maxValueSoFar, weight * value / maxValueSoFar, maxValueSoFar)

        # So: for the BCE loss
        # Calculate the loss
        discriminator_error_fake =  criterion(output, label)
        # Calculate the normalized weighted value and also get the new maximum
        bce_norm_loss, bce_norm_weighted_loss, loss_maxima[0] = normalizedAndWeightedLoss(discriminator_error_fake, loss_maxima[0], loss_weights[0])
        current_batch_loss += bce_norm_weighted_loss
        BCELoss += bce_norm_loss

        # So: for the NGG loss
        # Get the maximum
        if (loss_maxima[1] == 0.0):
            ngg_loss_normalizer = 1.0
        else:
            ngg_loss_normalizer = loss_maxima[1]
        # Calculate the loss
        generator_content_loss = content_loss_fn(fake.cpu().detach().numpy().squeeze())  # Get content-based-loss
        generator_content_loss = torch.mean(generator_content_loss).to(fake.device)
        ngg_loss = generator_content_loss.item() / len(dataloader) # Get average generator loss

        # and normalize by the maximum
        ngg_loss_norm = ngg_loss / ngg_loss_normalizer
        # Update the maximum so far
        # TODO: Make sure that this is returned to be used by the caller
        if loss_maxima[1] < ngg_loss:
            loss_maxima[1] = ngg_loss # Update maximum
        ngg_norm_weighted = loss_weights[1] * ngg_loss_norm
        current_batch_loss += ngg_norm_weighted
        NGramGraphLoss += ngg_loss_norm  # Update overall across batches

        # So: for the HistoFourier loss
        # Get the maximum
        if (loss_maxima[2] == 0.0):
            histo_fourier_loss_normalizer = 1.0
        else:
            histo_fourier_loss_normalizer = loss_maxima[2]
        # Calculate the loss
        generator_vector_content_loss = vector_content_loss_fn(fake.cpu().detach().numpy().squeeze())
        generator_vector_content_loss = torch.mean(generator_vector_content_loss).to(fake.device)
        histo_fourier_loss = generator_vector_content_loss.item() / len(dataloader)

        # and normalize by the maximum
        histo_fourier_loss_norm = histo_fourier_loss / histo_fourier_loss_normalizer
        # Update the maximum so far
        # TODO: Make sure that this is returned to be used by the caller
        if loss_maxima[2] < histo_fourier_loss_norm:
            loss_maxima[2] = histo_fourier_loss_norm # Update maximum
        histo_fourier_loss_norm_weighted = histo_fourier_loss_norm * loss_weights[2] # weight for HeightHistogramAndFourierLoss
        current_batch_loss += histo_fourier_loss_norm_weighted
        HeightHistogramAndFourierLoss += histo_fourier_loss_norm # Update overall across batches

        # Update overall_loss with batch contribution
        overall_loss += current_batch_loss

        # Calculate gradients for G, which propagate through the discriminator
        overall_loss.backward()
        discriminator_output_fake_batch = output.mean().item()
        # Update G
        optimizer_generator.step()

        # calculate total losses
        generator_loss += overall_loss.item() / len(dataloader)
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
        NGramGraphLoss,
        HeightHistogramAndFourierLoss,
    )
