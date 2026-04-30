import torch

from src.models.architectures.distributions.empirical import EmpiricalSampler

class EmpiricalBlockSampler:
    
    def __init__(self, cat_remv_process, dataset_info, device=None):
        
        self.cat_remv_process = cat_remv_process
        self.dataset_info = dataset_info
        self.empirical_sampler = EmpiricalSampler(dataset_info, device=device)
        
    
    def initialize(self, batch_size):
        # sample the number of nodes from the empirical distribution
        self._target_numnodes = self.empirical_sampler(batch_size=batch_size)
        self._current_numnodes = torch.zeros_like(self._target_numnodes)
        return self._target_numnodes
    
    def update(self, graphs, mask):
        self._target_numnodes = self._target_numnodes[mask]
        self._current_numnodes = graphs.num_nodes_per_sample
        
    def sample_block(self):
        
        return self.cat_remv_process.schedule.sample_nodes_to_add_posterior(
            n0=self._target_numnodes,
            nt=self._current_numnodes,
            t=None
        )
        
    def get_new_time(self, reversed_insertion_time):
        return self.cat_remv_process.normalize_time(
            t = reversed_insertion_time+1
        )
    
        
    def should_halt(self, graphs):
        return graphs.num_nodes_per_sample >= self._target_numnodes
        