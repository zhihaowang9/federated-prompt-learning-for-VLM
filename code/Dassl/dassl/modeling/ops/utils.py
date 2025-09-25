import numpy as np
import torch


def sharpen_prob(p, temperature=2):

    p = p.pow(temperature)
    return p / p.sum(1, keepdim=True)


def reverse_index(data, label):

    inv_idx = torch.arange(data.size(0) - 1, -1, -1).long()
    return data[inv_idx], label[inv_idx]


def shuffle_index(data, label):

    rnd_idx = torch.randperm(data.shape[0])
    return data[rnd_idx], label[rnd_idx]


def create_onehot(label, num_classes):
 
    onehot = torch.zeros(label.shape[0], num_classes)
    return onehot.scatter(1, label.unsqueeze(1).data.cpu(), 1)


def sigmoid_rampup(current, rampup_length):
   
    assert rampup_length > 0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current/rampup_length
    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):

    assert rampup_length > 0
    ratio = np.clip(current / rampup_length, 0.0, 1.0)
    return float(ratio)


def ema_model_update(model, ema_model, alpha):
    
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
