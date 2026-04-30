from typing import Optional, List, Union, Dict

import torch
from torch import Tensor
import torch.nn as nn
from src.models import reg_architectures

@reg_architectures.register()
class EmpiricalSampler(nn.Module):

    def __init__(
            self,
            dataset_info: Dict,
            device: Optional[str]=None,
            **kwargs
        ):
        super().__init__()

        self.output_type = 'regressor'

        # get the histogram of number of nodes
        nodes_hist = [(int(k), int(v)) for k, v in dataset_info['num_nodes_hist'].items()]
        nodes_hist = list(zip(*nodes_hist))
        nodes_idx = torch.tensor(nodes_hist[0], dtype=torch.int64, device=device)
        nodes_weights = torch.tensor(nodes_hist[1], dtype=torch.float, device=device)
        
        # e.g. histogram of the number of nodes in the dataset
        self.property_map = nn.Parameter(nodes_idx, requires_grad=False)
        self.property_histograms = nn.Parameter(nodes_weights, requires_grad=False)


    def forward(
            self,
            batch_size: Optional[int]=None,
            **kwargs
        ):

        # sample from the empirical distribution (multinomial)
        number_of_nodes_idx = torch.multinomial(
            input =         self.property_histograms,
            num_samples =   batch_size,
            replacement =   True
        )

        # retrieve the number of nodes from the property map
        number_of_nodes = self.property_map[number_of_nodes_idx]

        return number_of_nodes
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        key_map = prefix + 'property_map'
        key_histograms = prefix + 'property_histograms'
        if key_map in state_dict and key_histograms in state_dict:
            
            # the state might have different shapes, so we need to change
            # the shape of this module
            if self.property_map.shape != state_dict[key_map].shape:
                self.property_map = nn.Parameter(state_dict[key_map].clone().detach(), requires_grad=False)
                self.property_histograms = nn.Parameter(state_dict[key_histograms].clone().detach(), requires_grad=False)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)