##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/metrics/abstract_metrics.py
#
##########################################################################################################

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric
from torch_geometric.utils import to_dense_batch

import numpy as np
from scipy.stats import wasserstein_distance


def halting_prior_emd(
        pred_logits: Tensor,
        targets: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        max_seq_len: int
    ) -> Tensor:
    """ Compute EMD between predicted halting prior probability and true halting signal. """

    # compute batch from pred_logits and true
    if pred_logits.ndim == 1:
        pred_logits = pred_logits.unsqueeze(-1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)

    # here shapes are: (batch_size_flat, 1)
    # to_dense_batch requires that the batch is ordered
    ord = torch.argsort(batch_idx)
    batch_idx = batch_idx[ord].long()
    pred_logits = pred_logits[ord]
    targets = targets[ord]
    
    # compute dense batch
    pred_logits, _ = to_dense_batch(pred_logits, batch=batch_idx, max_num_nodes=max_seq_len, batch_size=batch_size)
    targets, mask =  to_dense_batch(targets,  batch=batch_idx, max_num_nodes=max_seq_len, batch_size=batch_size)

    pred_logits = pred_logits.squeeze(-1)
    targets = targets.squeeze(-1)
    # here shapes are: (batch_size, max_seq_len)

    # compute halting prior
    pred_probs = torch.sigmoid(pred_logits)

    # compute prior as prod(1 - p_j) * p_i
    neg_cumprod = torch.cumprod(1 - pred_probs, dim=-1)
    neg_cumprod = torch.cat([torch.ones_like(neg_cumprod[..., :1]), neg_cumprod[..., :-1]], dim=-1)
    prior = neg_cumprod * pred_probs

    # correct prior to sum to 1 (remaining mass is placed first to penalize)
    prior[..., 0] = prior[..., 0] + 1 - prior.sum(-1)
    prior = torch.clamp(prior, 0, 1)

    # compute EMD
    true_np = targets.cpu().numpy()
    prior_np = prior.cpu().numpy()
    values = np.arange(true_np.shape[-1])

    # compute pad mask
    mask = mask.cpu().numpy().astype(bool)

    # encapsulate in function for computing masked EMD
    fn_emd = lambda v, p_x, p_y: wasserstein_distance(v, v, p_x, p_y)
    fn_masked_emd = lambda v, p_x, p_y, m: fn_emd(v[m], p_x[m], p_y[m])

    # compute EMD
    emd = [fn_masked_emd(values, true_np[i], prior_np[i], mask[i]) for i in range(true_np.shape[0])]

    emd = torch.tensor(emd, device=pred_logits.device)

    return emd


class HaltingPriorEMD(Metric):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.add_state('total_emd', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred_logits: Tensor, targets: Tensor, batch_idx: Tensor, batch_size: int, max_seq_len: int) -> None:
        emd = halting_prior_emd(pred_logits, targets, batch_idx, batch_size, max_seq_len)
        
        self.total_emd += torch.sum(emd)
        if emd.ndim == 0:
            self.total_samples += 1
        else:
            self.total_samples += emd.shape[0]

    def compute(self):
        return self.total_emd / self.total_samples


class OpenClassesAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.ndim == 2:
            preds = preds.argmax(dim=-1)
        correct = torch.sum(preds == target)
        self.correct += correct
        self.total += preds.shape[0]

    def compute(self):
        return self.correct / self.total