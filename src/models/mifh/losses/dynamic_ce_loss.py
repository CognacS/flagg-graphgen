from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import src.models.ifh.labels as labels

from torch_geometric.nn import global_add_pool, global_max_pool


class DynamicCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, batch, ptr, reduce=True):
        # increment target index by ptr
        incr_target = target + ptr[:-1]

        # use the softmax-trick to avoid numerical instability
        # it happened!
        max_input = global_max_pool(input.detach(), batch, ptr.size(0) - 1)

        # compute log-sum-exp of dynamic shape
        input_exp = torch.exp(input - max_input[batch])
        sum_exp = global_add_pool(input_exp, batch, ptr.size(0) - 1)
        log_sum_exp = torch.log(sum_exp)

        # gather target input
        target_input = input[incr_target] - max_input

        # compute dynamic shape Cross Entropy
        loss = -target_input + log_sum_exp

        if reduce:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        else:
            return loss