from typing import List
import torch
from torch import Tensor

from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool


def group_unique(src, index):
    # Normalize `src` to range [0, 1]:
    src = src - src.min()
    src = src / src.max()

    # Compute the unique indices:
    src = src + 2 * index
    _, mapped_src = torch.unique(src, return_inverse=True)

    return mapped_src



def compute_ratio(w_v, prev_idx):
    w_avg = global_mean_pool(w_v, prev_idx)
    return w_v / (w_avg[prev_idx] + 1e-8)

def degap_idx(prev_idx, new_idx_noninc):
    # remove gaps from the index, and increment the index
    # prev_idx = [0, 0, 0, 1, 1, 2]
    # new_idx_noninc = [1, 2, 1, 1, 1, 0]
    # new_idx = [0, 1, 0, 4, 4, 5]
    new_idx = group_unique(new_idx_noninc, prev_idx)
    return new_idx

def update_idx(prev_idx, new_idx_noninc):
    # update the new index by incrementing its entries by
    # the cumulative number of nodes in the previous property sets.
    # this is needed to make the new index non-intersecting
    # between different property sets (from the old index)
    # e.g.
    # prev_idx = [0, 0, 0, 1, 1, 2]
    # new_idx_noninc = [1, 2, 1, 1, 1, 0]
    # ptr = [0, 3, 5, 6]
    # new_idx = [1, 2, 1, 4, 4, 5] = [1, 2, 1, 1+3, 1+3, 0+5]

    # compute the number of nodes in each property set
    num_nodes = torch.bincount(prev_idx)

    # compute the cumulative number of nodes
    ptr = torch.zeros((num_nodes.shape[0]+1,), dtype=torch.long) 
    ptr[1:] = torch.cumsum(num_nodes, dim=0)

    # return incremented index
    return new_idx_noninc + ptr[prev_idx]


def compute_probs(weights: Tensor, properties: List|Tensor, batch: Tensor):

    # initialize the previous index
    # the initial property set is the whole graph
    prev_idx = batch

    # initialize the node probabilities with a uniform prior distribution
    node_probs = torch.ones_like(batch, dtype=torch.float)
    # divide by the batch sizes
    node_probs /= global_add_pool(node_probs, batch)[batch]

    for w_v, idx in zip(weights[:-1], properties):
        
        # compute the new index
        #new_idx = update_idx(prev_idx, idx)
        new_idx = degap_idx(prev_idx, idx)

        # compute ratios
        w_ratio = compute_ratio(w_v, prev_idx)

        # update the node probabilities
        # this will be the posterior distribution after
        # observing the new property's value
        node_probs = node_probs * w_ratio

        # update the previous index
        prev_idx = new_idx

    # use the final weights: these are needed to give preference
    # to nodes inside the same property set
    # there is no need to compute a new index as the final property sets
    # are the single nodes

    # compute ratios
    w_ratio = compute_ratio(weights[-1], prev_idx)
    # update the node probabilities
    node_probs = node_probs * w_ratio

    return node_probs



def global_logmeanexp_pool(x, batch):
    x_max = global_max_pool(x, batch=batch)
    max_per_x = x_max.gather(-1, batch)
    x_exp_mean = global_mean_pool(torch.exp(x - max_per_x), batch=batch)
    return x_max + torch.log(x_exp_mean)

def global_logsumexp_pool(x, batch):
    x_max = global_max_pool(x, batch=batch)
    max_per_x = x_max.gather(-1, batch)
    x_exp_mean = global_add_pool(torch.exp(x - max_per_x), batch=batch)
    return x_max + torch.log(x_exp_mean)


def compute_logdiff(logw_v, prev_idx):
    lme = global_logmeanexp_pool(logw_v, prev_idx)
    return logw_v - lme[prev_idx]


def compute_logprobs(logweights: Tensor, properties: List|Tensor, batch: Tensor):

    # initialize the previous index
    # the initial property set is the whole graph
    prev_idx = batch

    # initialize the node log-probabilities with a uniform prior distribution
    ones = torch.ones_like(batch, dtype=torch.float)
    # divide by the batch sizes -> - log(batch_size)
    node_logprobs = - torch.log(global_add_pool(ones, batch)[batch])

    for logw_v, idx in zip(logweights[:-1], properties):
        
        # compute the new index
        #new_idx = update_idx(prev_idx, idx)
        new_idx = degap_idx(prev_idx, idx)

        # compute difference of logs
        logw_diff = compute_logdiff(logw_v, prev_idx)

        # update the node logprobabilities
        # this will be the posterior distribution after
        # observing the new property's value
        node_logprobs = node_logprobs + logw_diff

        # update the previous index
        prev_idx = new_idx

    # use the final weights: these are needed to give preference
    # to nodes inside the same property set
    # there is no need to compute a new index as the final property sets
    # are the single nodes

    # compute difference of logs
    logw_diff = compute_logdiff(logweights[-1], prev_idx)
    # update the node logprobabilities
    node_logprobs = node_logprobs + logw_diff

    return node_logprobs