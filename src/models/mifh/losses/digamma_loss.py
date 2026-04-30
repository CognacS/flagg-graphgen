from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import src.models.ifh.labels as labels
from torch_geometric.nn import global_add_pool

from src.models.mifh.batch_utils import batch_softmax


class DigammaLoss(nn.Module):

    def __init__(self, c=0.0):
        super().__init__()
        self.c = c

    def forward(self, input, target, total_hist, batch):
        # input expected to be the weights of a mvwnchypg distribution
        # target is a sampled histogram
        norm_const = (input * (total_hist - target + self.c))
        norm_const = global_add_pool(norm_const, batch)
        inv_norm_input = norm_const[batch] / input
        loss = (target * torch.special.digamma(1 + inv_norm_input))
        sums = global_add_pool(loss, batch)
        return sums.mean()



class DigammaWithLogitsLoss(nn.Module):

    def __init__(self, c=0.0):
        super().__init__()
        self.c = c

    def forward(self, input, target, total_hist, batch):
        # input expected to be the weights of a mvwnchypg distribution
        # target is a sampled histogram
        diff = torch.exp(input.unsqueeze(-2) - input.unsqueeze(-1))
        inv_norm_input = (diff * (total_hist.unsqueeze(-2) - target.unsqueeze(-2) + self.c)).sum(-1)
        loss = (target * torch.special.digamma(1 + inv_norm_input))
        sums = global_add_pool(loss, batch)
        return sums.mean()
    

class DigammaWithLogitsLoss(nn.Module):

    def __init__(self, c=0.0):
        super().__init__()
        self.c = c

    def forward(self, input, target, total_hist, batch, batch_size):
        # input expected to be the weights of a mvwnchypg distribution
        # target is a sampled histogram
        input = batch_softmax(input, batch, batch_size)

        norm_const = (input * (total_hist - target + self.c))
        norm_const = global_add_pool(norm_const, batch, size=batch_size)
        inv_norm_input = norm_const[batch] / (input + 1e-8)
        loss = (target * torch.special.digamma(1 + inv_norm_input))
        sums = global_add_pool(loss, batch, size=batch_size)
        return sums.mean()