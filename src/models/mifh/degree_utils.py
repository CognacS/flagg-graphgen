

from typing import List, Tuple

import torch
from torch import Tensor

from src.datatypes.sparse import SparseGraph, SparseEdges


def sparse_idx_to_mask(idx: Tensor, num_nodes: int):
    mask = torch.zeros(num_nodes, dtype=torch.float, device=idx.device)
    mask[idx] = 1.
    return mask


def batch_edges_to_adj(batch: SparseGraph, remv_edges_ba: SparseEdges):
    return sparse_idx_to_mask(remv_edges_ba.edge_index[1], batch.num_nodes)


from random import randint

def sample_sub_adj(degree: int, incr: int, device):
    num_keep = randint(0, max(degree-1, 0))
    randperm = torch.randperm(degree, device=device)
    keep_idx = randperm[:num_keep]
    keep_idx_next = randperm[:num_keep+1]
    return keep_idx + incr, keep_idx_next + incr



def batch_sample_sub_adj(
        batch: SparseGraph,
        nodes_index: Tensor,
        degrees: Tensor
    ):
    
    # sampled_idx contains the indices of the nodes to keep
    # where the number to keep is uniformly sampled in the range [0, degree-1]
    #TODO: possibility for an error here, have to check
    cum_degree = torch.cat([torch.zeros(1, device=degrees.device), torch.cumsum(degrees, dim=0)])
    sampled_idx = [sample_sub_adj(degree, cum_deg, device=degrees.device) for cum_deg, degree in zip(cum_degree, degrees)]
    sampled_idx, sampled_idx_next = zip(*sampled_idx)
    sampled_idx = torch.cat(sampled_idx)
    sampled_idx_next = torch.cat(sampled_idx_next)

    # extract the the nodes to keep
    sub_nodes_index = nodes_index[sampled_idx.to(dtype=torch.long)]
    sub_nodes_index_next = nodes_index[sampled_idx_next.to(dtype=torch.long)]

    # compute sub adjacency matrix
    sampled_sub_adj = sparse_idx_to_mask(sub_nodes_index, batch.num_nodes)
    sampled_sub_adj_next = sparse_idx_to_mask(sub_nodes_index_next, batch.num_nodes)

    return sampled_sub_adj, sampled_sub_adj_next


from src.models.mifh.distance_utils import subadj_by_avg_dist

def batch_sample_sub_adj_avg_dist(
        batch: SparseGraph,
        nodes_index: Tensor,
        degrees: Tensor
    ):
    
    n_keep = torch.Tensor([randint(0, max(degree-1, 0)) for degree in degrees]).to(device=degrees.device)

    adj = torch.zeros(batch.num_nodes, dtype=torch.float, device=degrees.device)
    adj[nodes_index] = 1

    # # print inputs
    # torch.set_printoptions(profile="full")
    # print("edge_index =", batch.edge_index)
    # print("adj =", adj)
    # print("num_nodes_per_sample =", batch.num_nodes_per_sample)
    # print("batch =", batch.batch)
    # print("remv_degree =", degrees)
    # print("num_to_sel =", n_keep)
    # print("threshold =", max(int(batch.num_nodes_per_sample.max().item() / 2),  5))
    # torch.set_printoptions(profile="default")

    # # be sure that prints have been flushed
    # import sys
    # sys.stdout.flush()

    sampled_sub_adj, sampled_sub_adj_next = subadj_by_avg_dist(
        edge_index=batch.edge_index,
        adj=adj,
        num_nodes_per_sample=batch.num_nodes_per_sample,
        batch=batch.batch,
        remv_degree=degrees,
        num_to_sel=n_keep,
        threshold=max(int(batch.num_nodes_per_sample.max().item() / 2),  5)#TODO: check this
    )

    return sampled_sub_adj, sampled_sub_adj_next



def create_fake_edge_index_batch(batch: Tensor, batch_size: int, undirected: bool=True):
    tot_num_nodes = batch.size(0)
    # create edge_index for the fake graph
    # bs = 3
    # arange+bs =   [3,4,5,6,7,8,9]
    # batch =       [0,0,0,1,1,2,2]
    # this will make existing nodes indexed from bs to bs+num_nodes-1
    # and the new nodes indexed from 0 to bs-1
    fake_edge_index = torch.stack([
        torch.arange(tot_num_nodes, device=batch.device) + batch_size,
        batch
    ])

    if undirected:
        fake_edge_index = torch.cat([
            fake_edge_index,
            fake_edge_index[[1,0]]
        ], dim=1)

    return fake_edge_index.to(dtype=torch.int64)