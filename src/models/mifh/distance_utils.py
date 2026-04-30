from typing import Optional

import torch
from torch import Tensor


def batched_nonzero(bool_tensor: Tensor, num_nodes_per_sample: Tensor):
    # get the indices of the non-zero elements in the boolean tensor
    nonzero_indices = torch.nonzero(bool_tensor, as_tuple=False).view(-1)

    # get the batch indices of the non-zero elements
    batch_indices = torch.repeat_interleave(torch.arange(num_nodes_per_sample.size(0), device=num_nodes_per_sample.device), num_nodes_per_sample)

    # get the indices of the non-zero elements in the boolean tensor
    return batch_indices[nonzero_indices], nonzero_indices


def _sharded_neighbors_rec(edge_index_0, bfs_queue, max_numel):
    estimated_numel = len(bfs_queue) * len(edge_index_0)
    # if the number of elements is too large, split the computation into 4 parts
    if estimated_numel > max_numel:
        cut_edge = edge_index_0.shape[0] // 2
        cut_queue = len(bfs_queue) // 2
        if cut_edge > 0:
            left = _sharded_neighbors_rec(edge_index_0[:cut_edge], bfs_queue[cut_queue:], max_numel)
        right = _sharded_neighbors_rec(edge_index_0[cut_edge:], bfs_queue[cut_queue:], max_numel)
        if cut_queue > 0:
            left += _sharded_neighbors_rec(edge_index_0[:cut_edge], bfs_queue[:cut_queue], max_numel)
            right += _sharded_neighbors_rec(edge_index_0[cut_edge:], bfs_queue[:cut_queue], max_numel)
        if cut_edge > 0:
            return torch.cat([left, right])
        else:
            return right
    else:
        # return the number of occurrences of each element in the BFS queue
        # if the size is not too large
        return (edge_index_0 == bfs_queue[:, None]).sum(-2)

def sharded_neighbors(edge_index, bfs_queue, max_numel=10000000):
    return edge_index[1, _sharded_neighbors_rec(edge_index[0], bfs_queue, max_numel).bool()]


def compute_distances(edge_index: Tensor, adj: Tensor, num_nodes_per_sample: Tensor, threshold: Optional[int]=None):
    # initialize the distances with -1
    distances = torch.full_like(adj, -1, dtype=torch.long, device=adj.device)
    distances[adj.bool()] = 0

    # initialize the BFS queue
    bfs_queue = torch.nonzero(adj, as_tuple=False).view(-1)
    #bfs_queue = adj.bool().clone()

    if threshold is None:
        threshold = num_nodes_per_sample.max().item()

    # iterate until the BFS queue is empty
    for i in range(threshold):

        # get the neighbors of the nodes in the BFS queue
        #neighbors = edge_index[1, torch.where(edge_index[0] == bfs_queue[:, None])[1]]
        #neighbors = edge_index[1, (edge_index[0] == bfs_queue[:, None]).sum(-2).bool()]
        neighbors = sharded_neighbors(edge_index, bfs_queue)
        #neighbors = edge_index[1, torch.nonzero(bfs_queue, as_tuple=False)[:, 0]]

        # get the nodes that are not already visited
        new_nodes = torch.where(distances[neighbors] == -1)[0]

        # update the distances
        distances[neighbors[new_nodes]] = i + 1

        # add the new nodes to the BFS queue
        bfs_queue = neighbors[new_nodes]

    return distances


from torch_geometric.utils import group_argsort

def subadj_by_avg_dist(edge_index, adj, num_nodes_per_sample, batch, remv_degree, num_to_sel, threshold):
    # compute distances from each node in adj to each other node in the graph
    
    # initialize the BFS queue
    bfs_queue = torch.nonzero(adj, as_tuple=False).view(-1)
    
    # normalize indices by batch
    ptr = torch.cat([torch.tensor([0], device=adj.device), num_nodes_per_sample.cumsum(dim=0)])
    bfs_queue_bidx = batch[bfs_queue]
    bfs_queue_inbatch = bfs_queue - ptr[bfs_queue_bidx]

    avg_distances = torch.full_like(adj, -1, dtype=torch.float)

    for i in range(remv_degree.max().item()):

        # select the nodes by order for each example
        bfs_queue_mask = bfs_queue_inbatch == i

        # prepare the mask for the current considered links
        curr_idx = bfs_queue[bfs_queue_mask]
        curr_adj = torch.zeros_like(adj)
        curr_adj[curr_idx] = 1

        # compute distances
        curr_distances = compute_distances(edge_index, curr_adj, num_nodes_per_sample, threshold)
        others_adj = (adj - curr_adj).bool()
        alive_mask = others_adj & (remv_degree[batch] > i)
        curr_distances = curr_distances[alive_mask]
        curr_distances[curr_distances == -1] = threshold


        # update the average distances
        idx_map = torch.zeros_like(remv_degree)
        idx_map[batch[curr_idx]] = torch.arange(curr_idx.size(0), device=adj.device)
        avg_distances.scatter_reduce_(0, curr_idx[idx_map[batch[alive_mask]]], curr_distances.float(), reduce='mean', include_self=False)

    sorted = group_argsort(avg_distances[adj.bool()], batch[adj.bool()])

    new_adj = sorted < num_to_sel.repeat_interleave(remv_degree)
    next_adj = sorted < (num_to_sel+1).repeat_interleave(remv_degree)
    next_adj = next_adj & ~new_adj

    full_adj = torch.zeros_like(adj)
    full_adj[bfs_queue[new_adj]] = 1

    full_next_adj = torch.zeros_like(adj)
    full_next_adj[bfs_queue[next_adj]] = 1

    return full_adj, full_next_adj