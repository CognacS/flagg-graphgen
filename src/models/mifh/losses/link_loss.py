from typing import List

import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool

class LinkLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, w_nod, target, batch, num_nodes):

        # compute normalizing factors
        num_targets = global_add_pool(target, batch) + 1e-8
        num_targets = num_targets.repeat_interleave(num_nodes)

        # compute loss, which is a Cross Entropy with a
        # uniform distribution
        logs = torch.log(w_nod + 1e-8)
        
        loss = - logs * target / num_targets
        loss = global_add_pool(loss, batch)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class LinkLossWithLogprobs(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, w_nod, target, batch, num_nodes, reduce=True):

        # compute normalizing factors
        num_targets = global_add_pool(target, batch) + 1e-8
        num_targets = num_targets.repeat_interleave(num_nodes)

        # compute loss, which is a Cross Entropy with a
        # uniform distribution
        logs = w_nod
        # replace -inf with 0
        logs[logs == float('-inf')] = 0
        
        loss = - logs * target / num_targets
        loss = global_add_pool(loss, batch)

        if reduce:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        else:
            return loss