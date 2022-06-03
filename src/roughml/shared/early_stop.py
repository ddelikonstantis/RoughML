

def early_stopping(generator_loss, patience, delta):
    """The following function aims to stop the training procedure when generator
    loss shows no significant change after a predefined consecutive number of epochs.
    
    patience: number of consecutive epochs where generator loss shows no significant change.
    delta: generator loss threshold.
    """
    early_stop = False
    loss_change = []
    loss_change.append(generator_loss)
    cntr, i = 0, 1
    for i in range(1, len(loss_change)):
        if (loss_change[i] > (loss_change[i-1] + delta)) or (loss_change[i] < (loss_change[i-1] - delta)):
            cntr = 0
        else:
            cntr += 1

    if cntr >= patience:
        early_stop = True


    return early_stop, i