

def early_stopping(gen_loss_hist, patience, delta):
    """The following function aims to stop the training procedure when generator
    loss shows no significant change after a predefined consecutive number of epochs.
    
    patience: number of consecutive epochs where generator loss shows no significant change.
    delta: generator loss threshold.
    """
    
    early_stop = False
    cntr, i = 0, 1
    for i in range(1, len(gen_loss_hist)):
        if (gen_loss_hist[i] > (gen_loss_hist[i-1] + delta)) or (gen_loss_hist[i] < (gen_loss_hist[i-1] - delta)):
            cntr = 0
        else:
            cntr += 1

    if cntr >= patience:
        early_stop = True


    return early_stop, i