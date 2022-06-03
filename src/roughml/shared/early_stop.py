

def early_stop(gen_loss, patience):
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