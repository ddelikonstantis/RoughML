import logging
import time

import torch

from roughml.content.loss import NGramGraphContentLoss

logger = logging.getLogger(__name__)


# Returns a normalized and a normalized weighted value of a measure. Normalization occurs before weighting.
# The function returns both the above value, but also the new max value so far.
def normalizedAndWeightedLoss(value, weight, max_value_so_far):
    # Update maximum
    if max_value_so_far < value:
        max_value_so_far = value
    # normalize value
    value_norm = value / max_value_so_far
    # normalize weighted value
    value_norm_weighted = weight * value / max_value_so_far

    # Return normalized value, normalized weighted value, and max value so far
    return value_norm, value_norm_weighted, max_value_so_far


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
    loss_maxima=[0.0, 0.0, 0.0, 0.0],
    log_every_n=None,
    load_checkpoint = None,
):
    generator.train()

    current_batch_loss = 0
    generator_loss, discriminator_loss = 0, 0
    discriminator_output_real, discriminator_output_fake = 0, 0
    BCELoss, NGramGraphLoss, HeightHistogramAndFourierLoss = 0, 0, 0

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
        # normalize total discriminator loss
        # Update maximum, loss_maxima[4] is maximum value for discriminator loss so far
        if loss_maxima[4] < discriminator_error_total:
            loss_maxima[4] = discriminator_error_total
        # normalize discriminator loss
        discriminator_error_total /= loss_maxima[4]
        # Update D
        optimizer_discriminator.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)

        # Calculate G's loss
        
        # Initialize overall loss to zero
        overall_loss = 0.0
        # For each loss component
        # Normalize it based on the maximum value we have seen in this loss
        # loss_maxima[0] is Binary Cross-Entropy maximum value so far
        # loss_maxima[1] is NGramGraphLoss maximum value so far
        # loss_maxima[2] is HeightHistogramAndFourierLoss value maximum so far
        # loss_maxima[3] is Discriminator loss value maximum so far
        # loss_weights[0] is Binary Cross-Entropy loss weight
        # loss_weights[1] is NGramGraphLoss weight
        # loss_weights[2] is HeightHistogramAndFourierLoss weight
        # Weight it and add it to the overall loss

        # So: for the BCE loss
        discriminator_error_fake = criterion(output, label) # get Binary Cross-Entropy loss
        # Calculate the normalized weighted value and also get the new maximum
        bce_norm_loss, bce_norm_weighted_loss, loss_maxima[0] = normalizedAndWeightedLoss(discriminator_error_fake, loss_weights[0], loss_maxima[0])
        current_batch_loss += bce_norm_weighted_loss # Update overall loss
        
        # So: for the N-Gram Graph loss
        generator_content_loss = content_loss_fn(fake.cpu().detach().numpy().squeeze())  # Get content-based-loss
        generator_content_loss = torch.mean(generator_content_loss).to(fake.device) # get average loss value
        ngg_loss = generator_content_loss
        # Calculate the normalized weighted value and also get the new maximum
        ngg_norm_loss, ngg_norm_weighted_loss, loss_maxima[1] = normalizedAndWeightedLoss(ngg_loss, loss_weights[1], loss_maxima[1])
        current_batch_loss += ngg_norm_weighted_loss # Update overall loss
        
        # So: for the Height Histogram And Fourier loss
        generator_vector_content_loss = vector_content_loss_fn(fake.cpu().detach().numpy().squeeze()) # get height histogram and fourier loss
        generator_vector_content_loss = torch.mean(generator_vector_content_loss).to(fake.device) # get average loss value
        histo_fourier_loss = generator_vector_content_loss
        # Calculate the normalized weighted value and also get the new maximum
        histo_fourier_norm_loss, histo_fourier_norm_weighted_loss, loss_maxima[2] = normalizedAndWeightedLoss(histo_fourier_loss, loss_weights[2], loss_maxima[2])
        current_batch_loss += histo_fourier_norm_weighted_loss # Update overall loss

        # Update overall_loss with batch contribution
        overall_loss += current_batch_loss

        # Calculate gradients for G, which propagate through the discriminator
        overall_loss.backward()
        discriminator_output_fake_batch = output.mean().item()
        # Update G
        optimizer_generator.step()

        # calculate total losses for this batch
        generator_loss += overall_loss.item() / len(dataloader) # update overall loss for this batch
        BCELoss += bce_norm_loss.item() / len(dataloader) # update Binary Cross-Entropy loss for this batch
        NGramGraphLoss += ngg_norm_loss.item() / len(dataloader) # update N-Gram Graph loss for this batch
        HeightHistogramAndFourierLoss += histo_fourier_norm_loss.item() / len(dataloader) # update Height Histogram And Fourier loss for this batch
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
        BCELoss, 
        loss_maxima
    )
