from typing import Tuple, List, Dict, Any

import math
import random

import numpy as np

import torch
from torch import Tensor, IntTensor, BoolTensor, FloatTensor

from src.noise import reg_diffusion, reg_schedule
from src.noise.core import TimeSampler, NoiseSchedule, NoiseProcess
from src.datatypes.split import get_subgraph, get_subgraph_splits
from src.datatypes.sparse import SparseGraph, SparseEdges, create_empty_graph

try:
    _GRAPHOPS_C_IMPORTED = True
    import src.datatypes.graphops.graphops_c as graphops
except (ImportError, ModuleNotFoundError):
    _GRAPHOPS_C_IMPORTED = False
    import src.datatypes.graphops.graphops_p as graphops

################################################################################
#                              UTILITY FUNCTIONS                               #
################################################################################

class RemovalProcessException(Exception):
    pass


def sample_single_bernoulli_mask(num_nodes: int, success_prob: float|FloatTensor):
    """
    sample node boolean mask, where each element is an independent bernolli
    each element has stay_probability probability of being removed
    """
    if isinstance(success_prob, Tensor):
        device = success_prob.device
    else:
        device = None

    mask = torch.rand((num_nodes.item(),), device=device).uniform_() < success_prob

    return mask


def sample_bernoulli_mask(num_nodes: int|IntTensor, success_prob: float|FloatTensor) -> BoolTensor:
    if isinstance(num_nodes, int):
        num_nodes = IntTensor([num_nodes])

    if isinstance(success_prob, float) or success_prob.ndim == 0:
        success_prob = [success_prob]

    mask = torch.cat([
        sample_single_bernoulli_mask(nodes_n, succ_p) for nodes_n, succ_p in zip(num_nodes, success_prob)
    ])
    
    return mask


################################################################################
#                          REMOVAL PROCESS SCHEDULES                           #
################################################################################

def clamp_probability(prob: float|FloatTensor) -> float|FloatTensor:
    if isinstance(prob, float):
        return min(max(prob, 0), 1)
    else:
        return torch.clamp(prob, min=0, max=1)

#########################  HYPERBOLIC REMOVAL SCHEDULE  ########################
@reg_schedule.register('hyperbolic_removal')
class HyperbolicRemovalSchedule(NoiseSchedule):

    def __init__(self, offset):
        super().__init__()
        # offset = a
        self.offset = offset

    def params_next(self, t, **kwargs):
        # q = 1/(t+a)
        # if a=1: q = 1/(t+1)
        return clamp_probability(1./(t + self.offset))

    def params_time_t(self, t, **kwargs):
        # pi = a/(t+a) = a*q
        # if a=1: pi = 1/(t+1) = q
        return clamp_probability(self.offset/(t + self.offset))

    def params_posterior(self, t, **kwargs):
        # q' = a/[t * (t+a-1)]
        # if a=1: q' = 1/t^2
        return clamp_probability(self.offset/(t * (t + self.offset - 1.)))

    def get_max_time(self, **kwargs):
        return None


##########################  LINEAR REMOVAL SCHEDULE  ###########################
@reg_schedule.register('linear_removal')
class LinearRemovalSchedule(NoiseSchedule):

    def __init__(self, max_time):
        super().__init__()
        # max_time = T
        self.max_time = max_time

    def params_next(self, t, **kwargs):
        # q = 1/(T+1-t)
        return clamp_probability(1./(self.max_time + 1 - t))

    def params_time_t(self, t, **kwargs):
        # pi = 1 - t/T
        return clamp_probability(1. - t / self.max_time)

    def params_posterior(self, t, **kwargs):
        # q'= 1/t
        return clamp_probability(1. / t)

    def get_max_time(self, **kwargs):
        return self.max_time

    def reverse_step(self, t, **kwargs):
        return self.max_time - t


######################  ADAPTIVE LINEAR REMOVAL SCHEDULE  ######################
@reg_schedule.register('adaptive_linear_removal')
class AdaptiveLinearRemovalSchedule(NoiseSchedule):

    def __init__(self, velocity):
        super().__init__()
        # velocity = v
        self.velocity = velocity

    def _compute_max_time(self, n0):
        do_round = torch.ceil if isinstance(n0, Tensor) else math.ceil
        return do_round(n0 / self.velocity)

    def params_next(self, t, **kwargs):
        # q = 1/(n_0+1-t)
        return clamp_probability(1./(self._compute_max_time(kwargs['n0']) + 1 - t))

    def params_time_t(self, t, **kwargs):
        # pi = 1 - t/n_0
        return clamp_probability(1. - t / self._compute_max_time(kwargs['n0']))

    def params_posterior(self, t, **kwargs):
        # q'= 1/t
        return clamp_probability(1. / t)

    def get_max_time(self, **kwargs):
        return self._compute_max_time(kwargs['n0'])

    def reverse_step(self, t, **kwargs):
        return torch.clamp(self._compute_max_time(kwargs['n0']) - t, min=0)


######################  ONE-SHOT REMOVAL SCHEDULE  ######################
@reg_schedule.register('one_shot_removal')
class OneShotRemovalSchedule(NoiseSchedule):
    """This class is meant as a placeholder, it doesn't
    do anything, and is used together with the
    OneShotRemovalProcess"""

    def __init__(self):
        super().__init__()

    def params_next(self, t, **kwargs):
        return None

    def params_time_t(self, t, **kwargs):
        return None

    def params_posterior(self, t, **kwargs):
        return None

    def get_max_time(self, **kwargs):
        return 1

    def reverse_step(self, t, **kwargs):
        return 1 - t

################################################################################
#                            NODES NUMBER SCHEDULERS                           #
################################################################################

######################  ADAPTIVE LINEAR REMOVAL SCHEDULE  ######################
class NodesNumberSchedule(NoiseSchedule):

    def sample_nodes_to_remove_next(self, n0, nt, t):
        raise NotImplementedError
    def sample_nodes_to_remove_time_t(self, n0, t):
        raise NotImplementedError
    def sample_nodes_to_add_posterior(self, n0, nt, t):
        raise NotImplementedError


    def params_next(self, t, **kwargs):
        n0 = kwargs['n0']
        nt = kwargs['nt']
        gt = self.sample_nodes_to_remove_next(n0, nt, t)
        return nt - gt

    def params_time_t(self, t, **kwargs):
        n0 = kwargs['n0']
        sum_gt = self.sample_nodes_to_remove_time_t(n0, t)
        return n0 - sum_gt

    def params_posterior(self, t, **kwargs):
        nt = kwargs['nt']
        n0 = kwargs['n0']
        bar_gt = self.sample_nodes_to_add_posterior(n0, nt, t)
        return bar_gt
    
    def reverse_step(self, t, **kwargs):
        max_time = self.get_max_time(**kwargs)
        if max_time is None:
            return None
        else:
            return max_time - t

@reg_schedule.register('deterministic_removal')
class DeterministicNodesSchedule(NodesNumberSchedule):

    def __init__(self, block_size):
        super().__init__()

        self.block_size = block_size

    def _compute_max_time(self, n0):
        do_round = torch.ceil if isinstance(n0, Tensor) else math.ceil
        return do_round(n0 / self.block_size)
    
    def _clamp_nodes(self, curr_nodes, remv_nodes):
        return torch.clamp(remv_nodes, max=curr_nodes)
    
    def sample_nodes_to_remove_next(self, n0, nt, t):
        return self._clamp_nodes(nt, self.block_size)
    def sample_nodes_to_remove_time_t(self, n0, t):
        return self._clamp_nodes(n0, self.block_size * t)
    def sample_nodes_to_add_posterior(self, n0, nt, t):
        raise self._clamp_nodes(n0 - nt, self.block_size)


    def get_max_time(self, **kwargs):
        return self._compute_max_time(kwargs['n0'])

@reg_schedule.register('binomial_removal')
class BinomialNodesSchedule(NodesNumberSchedule):

    def __init__(self, binomial_schedule_type, **binomial_params):
        super().__init__()

        self.binomial_schedule = resolve_removal_schedule(
            binomial_schedule_type
        )(**binomial_params)


    def _sample_nodes(self, n, prob):
        return torch.distributions.binomial.Binomial(
            total_count = n,
            probs = prob
        ).sample()
    
    def sample_nodes_to_remove_next(self, n0, nt, t):

        # get removal probability from t-1 to t
        removal_prob = self.binomial_schedule.params_next(t, n0=n0, nt=nt)

        return self._sample_nodes(nt, removal_prob)
    
    def sample_nodes_to_remove_time_t(self, n0, t):
        
        # get surv probability from 0 to t
        surv_prob = self.binomial_schedule.params_time_t(t, n0=n0)

        return self._sample_nodes(n0, 1 - surv_prob)


    def sample_nodes_to_add_posterior(self, n0, nt, t):
            
        # get removal probability from 0,t to t-1
        post_prob = self.binomial_schedule.params_posterior(t, n0=n0, nt=nt)

        return self._sample_nodes(n0 - nt, post_prob)


    def get_max_time(self, **kwargs):
        return self.binomial_schedule.get_max_time(**kwargs)

from src.noise.utils import CoinChange, DistCoinChange
from scipy.stats import multivariate_hypergeom

@reg_schedule.register('categorical_removal')
class CategoricalNodesSchedule(NodesNumberSchedule):

    def __init__(self, block_size_options: List[int], precompute_size: int = 100, dist_type: str = 'histogram'):
        super().__init__()

        self.coin_change_dist = DistCoinChange(block_size_options, precompute_size, dist_type=dist_type)


    @property
    def categories(self):
        return self.coin_change_dist.coins_map


    def _get_histogram(self, n):
        if isinstance(n, int):
            # get removal probability from t-1 to t
            remv_block_size_counts = self.coin_change.get_coins_histogram(n)
        else:
            remv_block_size_counts = [
                self.coin_change.get_coins_histogram(int(n_i)) for n_i in n
            ]
        return remv_block_size_counts
    
    
    def _normalize_histogram(self, hist):
        return hist / hist.sum(dim=-1, keepdim=True)


    def _sample_nodes_categorical(self, n):

        device = None if not isinstance(n, Tensor) else n.device

        remv_block_size_count = torch.tensor(self._get_histogram(n), device=device)
        remv_block_size_dist = self._normalize_histogram(remv_block_size_count)

        return self.sample_nodes_from_dist(probs=remv_block_size_dist, device=device)
    
    
    def sample_nodes_from_dist(self, probs=None, logits=None):

        return self.coin_change_dist.sample_categorical(probs=probs, logits=logits)

    

    def _sample_nodes_multivariate_hypergeometric(self, n0, t):

        remv_block_size_count = np.array(self._get_histogram(n0), dtype=np.int32)
        device = None
        if isinstance(t, Tensor):
            device = t.device
            t = t.cpu().numpy().astype(np.int32)

        sampled_categories = multivariate_hypergeom.rvs(
            m = remv_block_size_count,   # number of balls in the urn for each color
            n = t                       # extract t balls from the urn
        )
        sampled_categories = torch.from_numpy(sampled_categories).to(device=device, dtype=torch.float)

        weights = torch.tensor(self.coin_change.get_coins_map(), dtype=torch.float, device=device)
        sampled_nodes = torch.inner(sampled_categories, weights)

        return sampled_nodes.int()
    

    def prepare_data(self, datapoint: SparseGraph, **kwargs):
        num_nodes = datapoint.num_nodes_per_sample
        if isinstance(num_nodes, Tensor):
            num_nodes = num_nodes.max().item()
        self.coin_change_dist.update_histograms(num_nodes)

    
    def sample_nodes_to_remove_next(self, n0, nt, t):
        return self.coin_change_dist.sample_categorical_from_amounts(nt, safe=False)

    
    def sample_nodes_to_remove_time_t(self, n0, t):
        return self.coin_change_dist.sample_multivariate_hypergeometric(n0, t, safe=False)


    def sample_nodes_to_add_posterior(self, n0, nt, t):
        return self.coin_change_dist.sample_categorical_from_amounts(n0 - nt, safe=False, reverse=True)
    
    
    def get_posterior_distribution(self, n0, nt, t):

        block_size_dist = self.coin_change_dist(n0 - nt, safe=False, reverse=True)

        return block_size_dist


    def get_max_time(self, **kwargs):
        if 'normalize' in kwargs and kwargs['normalize']:
            return None # do not normalize time!
        
        if 'n0' in kwargs:
            n0 = kwargs['n0']
            return self.coin_change_dist.get_coins_used(n0)
        else:
            return None


################################################################################
#                              REMOVAL PROCESSES                               #
################################################################################

@reg_diffusion.register('random_masking')
class RandomMaskingRemovalProcess (NoiseProcess):

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            batch_size: int,
            initialization: Dict[str, Tensor],
            device=None
        ) -> SparseGraph:

        return create_empty_graph(batch_size, initialization, device=device)

    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        # get removal probability
        removal_prob = self.get_params_next(t, **kwargs)

        # get number of nodes
        num_nodes = current_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            # sample a binomially distributed quantity
            num_alive_nodes = torch.distributions.binomial.Binomial(
                total_count = num_nodes,
                probs = 1 - removal_prob
            ).sample()

            return num_alive_nodes
        
        else:

            # sample bernoulli mask as noise
            mask = sample_bernoulli_mask(num_nodes, 1 - removal_prob)

            return mask


    def apply_noise_next(
            self,
            current_datapoint: SparseGraph,
            noise: BoolTensor,
            t: int|torch.IntTensor,
            **kwargs
        ) -> SparseGraph|Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        if 'split' in kwargs and kwargs['split']:

            # split graph into two graphs and the intermediate edges
            next_graph_a, next_graph_b, edges_ab, edges_ba = get_subgraph_splits(
                graph =				current_datapoint,
                boolean_mask =		noise
            )

            return next_graph_a, next_graph_b, edges_ab, edges_ba
        
        else:

            # compute subgraph from the boolean mask on nodes (noise)
            next_datapoint = get_subgraph(
                graph =				current_datapoint,
                boolean_mask =		noise
            )

            return next_datapoint

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(self, original_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):
        
        # get alive probability
        alive_prob = self.get_params_from_original(t, **kwargs)

        # get number of nodes
        num_nodes = original_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            # sample a binomially distributed quantity
            num_alive_nodes = torch.distributions.binomial.Binomial(
                total_count = num_nodes,
                probs = alive_prob
            ).sample()

            return num_alive_nodes

        else:

            # sample bernoulli mask as noise
            mask = sample_bernoulli_mask(num_nodes, alive_prob)

            return mask
    
    
    def apply_noise_from_original(
            self,
            original_datapoint: SparseGraph,
            noise: BoolTensor,
            t: int|torch.IntTensor,
            **kwargs
        ) -> SparseGraph|Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        if 'split' in kwargs and kwargs['split']:

            # split graph into two graphs and the intermediate edges
            next_graph_a, next_graph_b, edges_ab, edges_ba = get_subgraph_splits(
                graph =				original_datapoint,
                boolean_mask =		noise
            )

            return next_graph_a, next_graph_b, edges_ab, edges_ba
        
        else:

            # compute subgraph from the boolean mask on nodes (noise)
            next_datapoint = get_subgraph(
                graph =				original_datapoint,
                boolean_mask =		noise
            )

            return next_datapoint
        
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(self, original_datapoint: SparseGraph|IntTensor, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        # get posterior removal probability
        post_prob = self.get_params_posterior(t, **kwargs)

        # get number of nodes
        curr_num_nodes = current_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            # sample a binomially distributed quantity
            num_rem_nodes = torch.distributions.binomial.Binomial(
                total_count = original_datapoint - curr_num_nodes,
                probs = post_prob
            ).sample().int()

            return num_rem_nodes
        
        else:

            return None
        

def oneshot_adapter(datapoint: SparseGraph, split: bool=False):
    batch_size = datapoint.num_graphs

    global_attributes = datapoint.extract_attributes(datapoint.get_all_other_attrs())
    empty_graph: SparseGraph = create_empty_graph(batch_size, global_attributes, device=datapoint.edge_index.device)

    x_dim = datapoint.x.shape[-1]
    e_dim = datapoint.edge_attr.shape[-1]

    empty_graph.to_onehot(x_dim, e_dim)

    if split:
        edges_ab = SparseEdges(
            edge_index = torch.empty((2, 0), dtype=torch.long, device=datapoint.edge_index.device),
            edge_attr = torch.empty((0, e_dim), dtype=torch.float, device=datapoint.edge_index.device),
            num_nodes_s = empty_graph.num_nodes_per_sample,
            num_nodes_t = datapoint.num_nodes_per_sample,
            num_nodes = datapoint.num_nodes_per_sample
        )

        edges_ba = SparseEdges(
            edge_index = torch.empty((2, 0), dtype=torch.long, device=datapoint.edge_index.device),
            edge_attr = torch.empty((0, e_dim), dtype=torch.float, device=datapoint.edge_index.device),
            num_nodes_s = datapoint.num_nodes_per_sample,
            num_nodes_t = empty_graph.num_nodes_per_sample,
            num_nodes = datapoint.num_nodes_per_sample
        )

        return empty_graph, datapoint, edges_ab, edges_ba
    else:
        return empty_graph

@reg_diffusion.register('oneshot')
class OneShotRemovalProcess (NoiseProcess):

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            batch_size: int,
            initialization: Dict[str, Tensor],
            device=None
        ) -> SparseGraph:

        return create_empty_graph(batch_size, initialization, device=device)

    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        # get number of nodes
        num_nodes = current_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            return torch.zeros_like(num_nodes)
        else:
            return None


    def apply_noise_next(
            self,
            current_datapoint: SparseGraph,
            noise: BoolTensor,
            t: int|torch.IntTensor,
            **kwargs
        ) -> SparseGraph|Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        return oneshot_adapter(current_datapoint, split=('split' in kwargs and kwargs['split']))
    

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(self, original_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):
        
        # get number of nodes
        num_nodes = original_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            return torch.zeros_like(num_nodes) if torch.max(t) == 1 else num_nodes
        else:
            return None
    
    
    def apply_noise_from_original(
            self,
            original_datapoint: SparseGraph,
            noise: BoolTensor,
            t: int|torch.IntTensor,
            **kwargs
        ) -> SparseGraph|Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        if torch.max(t) == 1:
            return oneshot_adapter(original_datapoint, split=('split' in kwargs and kwargs['split']))
        else:
            return original_datapoint
        
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(self, original_datapoint: SparseGraph|IntTensor, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            if isinstance(original_datapoint, SparseGraph):
                return original_datapoint.num_nodes_per_sample
            elif isinstance(original_datapoint, Tensor):
                return original_datapoint
            else:
                raise RemovalProcessException(f'Could not resolve original datapoint type: {type(original_datapoint)}')
        else:
            raise NotImplementedError
        

################################################################################
#                             PERMUTATION REMOVAL                              #
################################################################################

def batch_permutation(datapoint: SparseGraph, perm_fun, n):
    """Attention: not parallized yet!"""
    if hasattr(datapoint, 'ptr'):
        batch_perm = []
        other_attrs = {}
        for i, curr_n in enumerate(n):
            if isinstance(curr_n, Tensor):
                curr_n = curr_n.cpu().item()

            order = perm_fun(datapoint[i], curr_n)
            if isinstance(order, tuple):
                order, others = order
                for k, v in others.items():
                    if k not in other_attrs:
                        other_attrs[k] = [v]
                    else:
                        other_attrs[k].append(v)
            batch_perm.append(np.argsort(order))

        # concatenate
        batch_perm = np.concatenate(batch_perm)
        for k in other_attrs.keys():
            other_attrs[k] = np.concatenate(other_attrs[k])

        if len(other_attrs) > 0:
            batch_perm = (batch_perm, other_attrs)

    else:
        batch_perm = perm_fun(datapoint, n)

    return batch_perm


def random_permutation(datapoint: SparseGraph, n: int) -> np.array:
    return np.random.permutation(n).astype(np.int32)

GRAPHOPS_PERMUTATION_FUNS = {
    'BFS': graphops.get_bfs_order,
    'DFS': graphops.get_dfs_order
}

class GraphOpPermutation:

    def __init__(self, which='BFS'):
        if which not in GRAPHOPS_PERMUTATION_FUNS:
            raise ValueError(f'Unknown graphop permutation type: {which}, expected one of {list(GRAPHOPS_PERMUTATION_FUNS.keys())}')
        self.which = which

    def __call__(self, datapoint: SparseGraph, n: int):
        if hasattr(datapoint, 'ptr'):
            raise NotImplementedError(f'{self.which} permutation not implemented for batched graphs')
        #start = random.randint(0, n-1)
        start = 0
        # get a random permutation of the node indices
        random_perm = np.random.permutation(n).astype(np.int32)
        # map to new permutation
        edge_index = random_perm[datapoint.edge_index.cpu().numpy().astype(np.int32)]
        # get BFS ordering with this permutation
        order = GRAPHOPS_PERMUTATION_FUNS[self.which](edge_index, n, start)
        if isinstance(order, tuple):
            order, depth = order
        else:
            order, depth = order[0], order[1]
        # map back to original node indices
        map_back = np.argsort(random_perm)
        order = np.argsort(map_back[order]).astype(np.int32)
        # reorder depth to original order
        depth = depth[random_perm]

        return order, {'node_depth': depth}


def _exp_perm(degree, n, param):
    exponential_probs = param * (- param * degree).exp()
    order = exponential_probs.multinomial(num_samples=n)
    return np.argsort(order.cpu().numpy()).astype(np.int32)

def _causal_exp_perm(datapoint: SparseGraph, n: int, param: float, inverse: bool = False):
    degree = datapoint.indegree
    edge_index = datapoint.edge_index
    idx = np.zeros(n, dtype=np.int32)
    for i in range(n):
        exponential_probs = (- param * degree).exp()
        sel = exponential_probs.multinomial(1).squeeze().cpu().item()
        if inverse:
            idx[sel] = n - i - 1
        else:
            idx[sel] = i
        degree[sel] = float("Inf")
        links = edge_index[1, edge_index[0] == sel]
        degree.scatter_add_(0, links, -torch.ones(links.size(0)))

    return idx


def exp_permutation(datapoint: SparseGraph, n: int):
    degree = datapoint.indegree
    param = 1.
    return _exp_perm(degree, n, param)

def invexp_permutation(datapoint: SparseGraph, n: int):
    degree = datapoint.indegree
    degree = degree.max() - degree
    param = 1.
    return _exp_perm(degree, n, param)

def causal_exp_permutation(datapoint: SparseGraph, n: int):
    return _causal_exp_perm(datapoint, n, param=3., inverse=False)

def causal_invexp_permutation(datapoint: SparseGraph, n: int):
    return _causal_exp_perm(datapoint, n, param=3., inverse=True)


def mark_datapoint_with_perm(datapoint: SparseGraph, permutation: np.ndarray):
    other_attrs = {}
    if isinstance(permutation, tuple):
        permutation, other_attrs = permutation

    # index perm for each node is the number of iteration in which that node is visited
    # in the permutation order
    datapoint.node_perm = torch.from_numpy(permutation).to(datapoint.edge_index.device)

    # set all other attributes
    for k, v in other_attrs.items():
        datapoint[k] = torch.from_numpy(v).to(datapoint.edge_index.device)


def get_mask_survivors(datapoint: SparseGraph, num_survivors: int|IntTensor) -> np.ndarray:

    if hasattr(datapoint, 'ptr'):
        # create a tensor which, for each node in a sample of the batch
        # contains the number of survivors for that sample
        num_survivors = torch.repeat_interleave(num_survivors, datapoint.num_nodes_per_sample, dim=0)

    # in the batch case, each node index is compared to the number of survivors of the sample
    # in the single case, each node index is compared to the int number of survivors
    mask = datapoint.node_perm < num_survivors

    return mask


PERMUTATION_TYPE_RANDOM = 'random'
PERMUTATION_TYPE_BFS = 'bfs'
PERMUTATION_TYPE_DFS = 'dfs'
PERMUTATION_TYPE_EXP = 'exp'
PERMUTATION_TYPE_INVEXP = 'invexp'
PERMUTATION_TYPE_CAUSAL_EXP = 'causal_exp'
PERMUTATION_TYPE_CAUSAL_INVEXP = 'causal_invexp'

PERMUTATIONS = {
    PERMUTATION_TYPE_RANDOM: random_permutation,
    PERMUTATION_TYPE_BFS: GraphOpPermutation(which='BFS'),
    PERMUTATION_TYPE_DFS: GraphOpPermutation(which='DFS'),
    PERMUTATION_TYPE_EXP: exp_permutation,
    PERMUTATION_TYPE_INVEXP: invexp_permutation,
    PERMUTATION_TYPE_CAUSAL_EXP: causal_exp_permutation,
    PERMUTATION_TYPE_CAUSAL_INVEXP: causal_invexp_permutation
}

@reg_diffusion.register('permutation')
class PermutationRemovalProcess (RandomMaskingRemovalProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            permutation_type : str = PERMUTATION_TYPE_RANDOM
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """

        assert isinstance(schedule, NodesNumberSchedule), \
            'PermutationRemovalProcess requires a NodesNumberSchedule, not a standard NoiseSchedule'

        # call super for the NoiseProcess
        super().__init__(schedule=schedule)

        self.permutation_type = permutation_type

        if self.permutation_type == PERMUTATION_TYPE_BFS and not _GRAPHOPS_C_IMPORTED:
            print('Warning: graphops_c could not be imported for BFS permutations, using graphops_p instead (much slower)')


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        #num_survivors = super().sample_noise_next(current_datapoint, t, return_quantity=True, **kwargs)
        nt = current_datapoint.num_nodes_per_sample
        num_survivors = self.schedule.params_next(t=t, nt=nt, **kwargs)

        if 'return_quantity' in kwargs and kwargs['return_quantity']:
            
            return num_survivors.int()
        
        else:

            # mask removes those nodes which are next in the starting
            # permutation order
            mask = get_mask_survivors(current_datapoint, num_survivors)

            return mask
        
    def apply_noise_next(
            self, current_datapoint: SparseGraph, noise: BoolTensor, t: int | IntTensor, **kwargs
        ) -> SparseGraph | Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        #if current_datapoint.node_perm.shape[0] != current_datapoint.num_nodes:
        #    current_datapoint.node_perm = current_datapoint.node_perm[noise]
        res = super().apply_noise_next(current_datapoint, noise, t, **kwargs)
        # if isinstance(res, tuple):
        #     res[0].node_perm = current_datapoint.node_perm[noise]
        return res

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def prepare_data(self, datapoint: SparseGraph, **kwargs):
        super().prepare_data(datapoint, **kwargs)
        # get number of nodes
        num_nodes = datapoint.num_nodes_per_sample
        # get permutation for each sample in the batch
        permutation = batch_permutation(
            datapoint,
            PERMUTATIONS[self.permutation_type],
            num_nodes
        )

        # permanently mark the datapoint with the permutation, to be used during next transitions
        mark_datapoint_with_perm(datapoint, permutation)


    def sample_noise_from_original(self, original_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        if not hasattr(original_datapoint, 'node_perm'):
            self.prepare_data(original_datapoint, **kwargs)

        #num_survivors = super().sample_noise_from_original(original_datapoint, t, return_quantity=True, **kwargs)
        if 'n0' not in kwargs:
            kwargs['n0'] = original_datapoint.num_nodes_per_sample
        num_survivors = self.schedule.params_time_t(t=t, **kwargs)

        if 'return_quantity' in kwargs and kwargs['return_quantity']:

            return num_survivors.int()

        else:
            # mask removes those nodes which are next in the starting
            # permutation order
            mask = get_mask_survivors(original_datapoint, num_survivors)

            return mask
        
    def apply_noise_from_original(
            self, original_datapoint: SparseGraph, noise: BoolTensor, t: int | IntTensor, **kwargs
        ) -> SparseGraph | Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        #if original_datapoint.node_perm.shape[0] != original_datapoint.num_nodes:
        #    original_datapoint.node_perm = original_datapoint.node_perm[noise]

        res = super().apply_noise_from_original(original_datapoint, noise, t, **kwargs)
        # if isinstance(res, tuple):
        #     res[0].node_perm = original_datapoint.node_perm[noise]
        return res
        
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(self, original_datapoint: SparseGraph|IntTensor, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        #super().sample_noise_posterior(original_datapoint, current_datapoint, t, **kwargs)
        n0 = original_datapoint
        nt = current_datapoint.num_nodes_per_sample
        num_new_nodes = self.schedule.params_posterior(t=t, nt=nt, n0=n0, **kwargs)

        return num_new_nodes.int()
    



################################################################################
#                              CLUSTER DIFFUSION                               #
################################################################################

@reg_schedule.register('cluster')
class ClusterCategoricalSchedule(NodesNumberSchedule):

    def __init__(self, sizes=None):
        super().__init__()

        if sizes is not None:
            self.set_sizes(sizes)

    def set_sizes(self, sizes):
        if not hasattr(self, 'categories'):
            self.register_buffer('categories', torch.tensor(sizes, dtype=torch.int32), persistent=True)

    def apply_dataset_info(self, dataset_info: Dict[str, Any]):
        sizes_hist = dataset_info['sizes_hist']
        sizes = sorted([int(k) for k in sizes_hist.keys()])
        self.set_sizes(sizes)
    
    def sample_nodes_from_dist(self, probs=None, logits=None):

        sampled_categories = torch.distributions.categorical.Categorical(
            probs = probs,
            logits = logits
        ).sample()
        
        return self.categories[sampled_categories]
    
    def get_max_time(self, **kwargs):
        if 'normalize' in kwargs and kwargs['normalize']:
            return None # do not normalize time!
        
        if 'data' in kwargs:
            data = kwargs['data']
            return data.special_supergraph.num_nodes_per_sample
        else:
            return None


@reg_diffusion.register('cluster')
class ClusterRemovalProcess (RandomMaskingRemovalProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            permutation_type : str = PERMUTATION_TYPE_RANDOM
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """

        assert isinstance(schedule, ClusterCategoricalSchedule), \
            'PermutationRemovalProcess requires a ClusterCategoricalSchedule, not a standard NoiseSchedule'

        # call super for the NoiseProcess
        super().__init__(schedule=schedule)

        self.permutation_type = permutation_type

        if self.permutation_type == PERMUTATION_TYPE_BFS and not _GRAPHOPS_C_IMPORTED:
            print('Warning: graphops_c could not be imported for BFS permutations, using graphops_p instead (much slower)')


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        # get number of nodes
        nt = current_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:

            # get number of nodes in the supergraph
            nt_super = current_datapoint.special_supergraph.num_nodes_per_sample
            t_rep = torch.repeat_interleave(t, nt_super, dim=0)
            # get mask for alive nodes (true for alive) in the supergraph
            supgraph_mask = current_datapoint.special_supergraph.node_perm == t_rep
            # get number of nodes in the community to remove
            num_nodes_per_comm = current_datapoint.special_supergraph.x
            return nt - num_nodes_per_comm[supgraph_mask]
        
        else:

            # repeat t for each node
            t_rep = torch.repeat_interleave(t, nt, dim=0)

            # get mask for alive nodes (true for alive)
            mask = current_datapoint.node_perm >= t_rep

            return mask

        
    def apply_noise_next(
            self, current_datapoint: SparseGraph, noise: BoolTensor, t: int | IntTensor, **kwargs
        ) -> SparseGraph | Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        res = super().apply_noise_next(current_datapoint, noise, t, **kwargs)

        return res

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def prepare_data(self, datapoint: SparseGraph, **kwargs):
        super().prepare_data(datapoint, **kwargs)

        super_graph = datapoint.special_supergraph
        # get number of nodes
        num_nodes = super_graph.num_nodes_per_sample
        # get permutation for each super graph in the batch
        permutation = batch_permutation(
            super_graph,
            PERMUTATIONS[self.permutation_type],
            num_nodes
        )

        # apply permutation to each subgraph
        # in detail, each node has the index of the super node it is part of
        # then, give the node the same index as the super node
        # such that, in one step, all nodes of the partition are removed together

        istuple = isinstance(permutation, tuple)
        if istuple:
            permutation, other_attrs = permutation
        permutation = permutation[datapoint.node_partition.cpu().numpy()]

        # permanently mark the datapoint with the permutation, to be used during next transitions
        mark_datapoint_with_perm(datapoint, permutation)


    def sample_noise_from_original(self, original_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        if not hasattr(original_datapoint, 'node_perm'):
            self.prepare_data(original_datapoint, **kwargs)

        #num_survivors = super().sample_noise_from_original(original_datapoint, t, return_quantity=True, **kwargs)
        if 'n0' not in kwargs:
            kwargs['n0'] = original_datapoint.num_nodes_per_sample

        n0 = original_datapoint.num_nodes_per_sample

        if 'return_quantity' in kwargs and kwargs['return_quantity']:

            # get number of nodes in the supergraph
            n0_super = original_datapoint.special_supergraph.num_nodes_per_sample
            t_rep = torch.repeat_interleave(t, n0_super, dim=0)
            # get mask for alive nodes (true for alive) in the supergraph
            supgraph_mask = original_datapoint.special_supergraph.node_perm == t_rep
            # get number of nodes in the community to remove
            num_nodes_per_comm = original_datapoint.special_supergraph.x
            return n0 - num_nodes_per_comm[supgraph_mask]
        
        else:

            # repeat t for each node
            t_rep = torch.repeat_interleave(t, n0, dim=0)

            # get mask for alive nodes (true for alive)
            mask = original_datapoint.node_perm >= t_rep

            return mask


        
    def apply_noise_from_original(
            self, original_datapoint: SparseGraph, noise: BoolTensor, t: int | IntTensor, **kwargs
        ) -> SparseGraph | Tuple[SparseGraph, SparseGraph, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        res = super().apply_noise_from_original(original_datapoint, noise, t, **kwargs)

        return res
        
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(self, original_datapoint: SparseGraph|IntTensor, current_datapoint: SparseGraph, t: int|torch.IntTensor, **kwargs):

        n0 = original_datapoint
        # get number of nodes in the supergraph
        nt_super = current_datapoint.special_supergraph.num_nodes_per_sample
        t_rep = torch.repeat_interleave(t-1, nt_super, dim=0)
        # get mask for alive nodes (true for alive) in the supergraph
        supgraph_mask = current_datapoint.special_supergraph.node_perm == t_rep
        # get number of nodes in the community to remove
        num_nodes_per_comm = current_datapoint.special_supergraph.x

        return num_nodes_per_comm[supgraph_mask]

################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

REMOVAL_SCHEDULE_HYPERBOLIC = 'hyperbolic'
REMOVAL_SCHEDULE_LINEAR = 'linear'
REMOVAL_SCHEDULE_ADAPTIVE_LINEAR = 'adaptive_linear'
REMOVAL_SCHEDULE_ONESHOT = 'oneshot'

REMOVAL_NUMNODES_SCHEDULE_DETERMINISTIC = 'deterministic'
REMOVAL_NUMNODES_SCHEDULE_BINOMIAL = 'binomial'
REMOVAL_NUMNODES_SCHEDULE_CATEGORICAL = 'categorical'

REMOVAL_PROCESS_RANDOM_MASKING = 'random_masking'
REMOVAL_PROCESS_ONESHOT = 'oneshot'
REMOVAL_PROCESS_PERMUTATION = 'permutation'

REMOVAL_SCHEDULES = {
    REMOVAL_SCHEDULE_HYPERBOLIC: HyperbolicRemovalSchedule,
    REMOVAL_SCHEDULE_LINEAR: LinearRemovalSchedule,
    REMOVAL_SCHEDULE_ADAPTIVE_LINEAR: AdaptiveLinearRemovalSchedule,
    REMOVAL_SCHEDULE_ONESHOT: OneShotRemovalSchedule,
    REMOVAL_NUMNODES_SCHEDULE_DETERMINISTIC: DeterministicNodesSchedule,
    REMOVAL_NUMNODES_SCHEDULE_BINOMIAL: BinomialNodesSchedule,
    REMOVAL_NUMNODES_SCHEDULE_CATEGORICAL: CategoricalNodesSchedule
}

REMOVAL_PROCESSES = {
    REMOVAL_PROCESS_RANDOM_MASKING: RandomMaskingRemovalProcess,
    REMOVAL_PROCESS_ONESHOT: OneShotRemovalProcess,
    REMOVAL_PROCESS_PERMUTATION: PermutationRemovalProcess
}



def resolve_removal_schedule(name: str) -> type:
    if name in REMOVAL_SCHEDULES:
        return REMOVAL_SCHEDULES[name]
    else:
        raise RemovalProcessException(f'Could not resolve removal schedule name: {name}')

def resolve_removal_process(name: str) -> type:
    if name in REMOVAL_PROCESSES:
        return REMOVAL_PROCESSES[name]
    else:
        raise RemovalProcessException(f'Could not resolve removal process name: {name}')