import torch
from torch import Tensor

from torch_geometric.nn import global_add_pool, global_max_pool


def batch_softmax(input, index, dim_sizes, multiplier=None):
    # compute log-sum-exp of dynamic shape
    # use trick
    maxes = global_max_pool(input, index, size=dim_sizes)
    input_exp = torch.exp(input - maxes[index])
    if multiplier is not None:
        input_exp = input_exp * multiplier
    sum_exp = global_add_pool(input_exp, index, size=dim_sizes)
    return input_exp / (sum_exp[index]+1e-8)

def batch_softmax_simple(input, multiplier=None):
    if multiplier is not None:
        input = input + torch.log(multiplier)
    return torch.softmax(input, dim=-1)

def batch_softmax_md(input, index, dim_sizes, multiplier=None):
    # compute log-sum-exp of dynamic shape
    # use trick
    maxes = scatter_md(input, index, size=dim_sizes, reduce='max')
    indices = torch.split(index, 1, dim=-1)
    input_exp = torch.exp(input - maxes[indices])
    if multiplier is not None:
        input_exp = input_exp * multiplier
    sum_exp = scatter_md(input_exp, index, size=dim_sizes, reduce='sum')
    return input_exp / (sum_exp[indices]+1e-8)
    


def get_incr_tensor(input: Tensor, ptr: Tensor, num_nodes_per_sample: Tensor):
    ptr = ptr[:-1]
    return input.to(torch.int64) + ptr.repeat_interleave(num_nodes_per_sample) # (N,)


def batch_idx_to_hist(
        num_nodes: int,
        incr_idx: Tensor
    ):
    
    hist = torch.zeros(num_nodes, dtype=torch.float, device=incr_idx.device)
    hist.scatter_add_(0, incr_idx, torch.ones_like(incr_idx, dtype=torch.float))

    return hist

def scatter_md(src, index, dim_sizes, reduce='sum'):
    """Computes scatter over multiple dimensions by defining index as a tensor of shape (N, D) where N is the number
    of elements to scatter and D is the number of dimensions to scatter over. The function will scatter the elements
    of src over the dimensions defined by index. The size of the dimensions is defined by dim_sizes.
    """
    # if index is 1D, we can use the scatter function directly
    if index.dim() == 1:
        out = torch.zeros(dim_sizes[0], dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, index, src, reduce=reduce, include_self=False)
        return out
    
    # compute scaling factors for flattening the index
    dims = torch.tensor(dim_sizes, dtype=torch.long)
    cumprod = torch.ones(len(dim_sizes)+1)
    cumprod[:-1] = torch.cumprod(dims.flip(0), dim=-1).flip(0)
    mult_idx = index * cumprod[1:].unsqueeze(0)
    # flatten indices (from tensor to 1D)
    flat_idx = torch.sum(mult_idx, dim=-1).long()
    # compute scatter
    out = torch.zeros(torch.prod(dims), dtype=src.dtype, device=src.device)
    #out.scatter_add_(0, flat_idx, src.view(-1))
    out.scatter_reduce_(0, flat_idx, src.view(-1), reduce=reduce, include_self=False)
    # return the result reshaped
    return out.view(*dims)

def compute_hist(idx, shape):
    """Computes a histogram of indices idx with the given shape. The histogram is computed by scattering ones over the
    indices and then summing them up."""
    hist = scatter_md(torch.ones(idx.shape[0], dtype=int), idx, shape)
    return hist

